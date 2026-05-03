"""Model-based voltage-constrained hosting capacity analysis."""

import warnings

import numpy as np

import yadi.dss.model as model


class DSS_VC_HCA(model.DSS_Data):
    """Iterative model-based hosting-capacity computation under a voltage upper bound."""

    def __init__(
        self,
        redirects: str | list[str],
        v_max: float = 1.05,
        delta_kw_inj: float = 1,
        kw_inj_max: float = 500,
        max_hc_iter: int = 50,
        verbose: bool = True,
    ) -> None:
        super().__init__(redirects, verbose=verbose)

        self.v_max = v_max
        self.delta_kw_inj = delta_kw_inj
        self.kw_inj_max = kw_inj_max
        self.max_hc_iter = max_hc_iter

        self.nodes = self.dss.Circuit.YNodeOrder()
        self.n_nodes = len(self.nodes)
        self.pfs = np.ones(self.n_nodes)

    def get_iterative_hc(self, pfs: np.ndarray | None = None) -> np.ndarray:
        """Return the vector of maximum kW hosting capacities for each node under `v_max`."""
        if pfs is not None:
            self.pfs = pfs

        self.compile_dss()
        self.dss.Text.Command("solve")

        vpu_base = np.copy(self.dss.Circuit.AllBusMagPu())
        sub_overvoltage_idx = [i for i in range(len(vpu_base)) if vpu_base[i] < self.v_max]

        hc = np.zeros(self.n_nodes)
        for node_idx, node in enumerate(self.nodes):
            if vpu_base[node_idx] > 1.05:
                hc[node_idx] = 0
                continue
            for kw_inj in [
                -1 * float(i)
                for i in np.arange(1.0, float(self.kw_inj_max), float(self.delta_kw_inj))
            ]:
                self.__get_perturbed_nodal_voltages(
                    bus_name=node, phases=1, kw_inj=kw_inj, kvar_inj=0
                )
                vpu_perturbed = np.asarray(self.dss.Circuit.AllBusMagPu())
                if np.any(vpu_perturbed[sub_overvoltage_idx] > self.v_max):
                    hc[node_idx] = kw_inj
                    break
        return hc

    def __get_perturbed_nodal_voltages(self, bus_name, phases, kw_inj, kvar_inj):
        """Get a dictionary of perturbed nodal voltages after a P/Q injection."""
        self.compile_dss()
        self.dss.Text.Command("Set Controlmode = STATIC")
        self.__set_perturbed_injection(bus_name, phases, kw_inj, kvar_inj)
        self.dss.Text.Command("solve")
        return self.get_node_voltages()

    def __set_perturbed_injection(self, bus_name, phases, kw_inj, kvar_inj):
        """Place a perturbing injection on a bus of interest."""
        bus_name = str(bus_name)
        phases = str(phases)
        kw_inj = str(kw_inj)
        kvar_inj = str(kvar_inj)
        injection_name = bus_name + "_static_inj"
        err = self.dss.Text.Command(
            "New Load.{injection_name} Bus1={bus_name} Phases={phases} "
            "kW ={kw_inj} kvar={kvar_inj}".format(
                injection_name="load" + str(injection_name),
                bus_name=bus_name,
                phases=phases,
                kw_inj=kw_inj,
                kvar_inj=kvar_inj,
            )
        )
        if self.verbose:
            print(err)
        elif err != "":
            warnings.warn("Perturbed injection failed, OpenDSS returned:")
            print(err)
