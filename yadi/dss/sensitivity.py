"""Perturb-and-observe sensitivity matrices for an OpenDSS network."""

import warnings

import numpy as np

import yadi.dss.model as model

SensitivityResult = dict[str, object]


class DSS_Sensitivities(model.DSS_Data):
    """Compute perturb-and-observe sensitivity matrices on top of `DSS_Data`."""

    def __init__(
        self,
        redirects: str | list[str],
        verbose: bool = True,
        per_unit: bool = True,
    ) -> None:
        super().__init__(redirects, verbose=verbose)
        self.spv: SensitivityResult | None = None
        self.sqv: SensitivityResult | None = None
        self.spth: SensitivityResult | None = None
        self.sqth: SensitivityResult | None = None
        self.per_unit = per_unit

    def get_svp(self) -> SensitivityResult:
        """Sensitivity of voltage magnitude with respect to active power."""
        spv, nodes, vph_base = self.__calc_sens_mat(indp_var="vmag", dep_var="p")
        self.spv = {"matrix": spv, "nodes": nodes, "vph_base": vph_base}
        return self.spv

    def get_svq(self) -> SensitivityResult:
        """Sensitivity of voltage magnitude with respect to reactive power."""
        sqv, nodes, vph_base = self.__calc_sens_mat(indp_var="vmag", dep_var="q")
        self.sqv = {"matrix": sqv, "nodes": nodes, "vph_base": vph_base}
        return self.sqv

    def get_sthp(self) -> SensitivityResult:
        """Sensitivity of voltage angle with respect to active power."""
        spth, nodes, vph_base = self.__calc_sens_mat(indp_var="theta", dep_var="p")
        self.spth = {"matrix": spth, "nodes": nodes, "vph_base": vph_base}
        return self.spth

    def get_sthq(self) -> SensitivityResult:
        """Sensitivity of voltage angle with respect to reactive power."""
        sqth, nodes, vph_base = self.__calc_sens_mat(indp_var="theta", dep_var="q")
        self.sqth = {"matrix": sqth, "nodes": nodes, "vph_base": vph_base}
        return self.sqth

    def __calc_sens_mat(self, indp_var="vmag", dep_var="p"):
        """Compute a perturb-and-observe sensitivity matrix.

        indp_var: independent variable -- "vmag" or "theta".
        dep_var: dependent variable -- "p" or "q".
        """
        self.compile_dss()
        self.dss.Text.Command("solve")

        nodes = self.dss.Circuit.YNodeOrder()

        if self.per_unit:
            vph_base = self.get_node_voltages_mag_pu()
        else:
            vph_base = self.get_node_voltages()
        vph_base_yorder = [vph_base[i] for i in nodes]

        S_matrix = np.zeros((len(nodes), len(nodes)))

        for col_idx, node in enumerate(nodes):
            if self.verbose:
                print("Node Perturbed: ", node)
            if dep_var in ("P", "p"):
                vph_perturbed = self.__get_perturbed_nodal_voltages(
                    bus_name=node, phases=1, kw_inj=-100, kvar_inj=0
                )
            elif dep_var in ("Q", "q"):
                vph_perturbed = self.__get_perturbed_nodal_voltages(
                    bus_name=node, phases=1, kw_inj=0, kvar_inj=-100
                )
            else:
                raise ValueError(f"Invalid dependent variable: {dep_var}")
            vph_perturbed_yorder = [vph_perturbed[i] for i in nodes]

            if indp_var == "vmag":
                S_matrix[:, col_idx] = (
                    np.abs(np.asarray(vph_perturbed_yorder)) - np.abs(np.asarray(vph_base_yorder))
                ) / (100)
            elif indp_var == "theta":
                S_matrix[:, col_idx] = (
                    np.angle(np.asarray(vph_perturbed_yorder))
                    - np.angle(np.asarray(vph_base_yorder))
                ) / (100)
            else:
                raise ValueError(f"Invalid independent variable: {indp_var}")

        return S_matrix, nodes, vph_base

    def __get_perturbed_nodal_voltages(self, bus_name, phases, kw_inj, kvar_inj):
        """Get a dictionary of perturbed nodal voltages after a P/Q injection."""
        self.compile_dss()
        self.dss.Text.Command("Set Controlmode = STATIC")
        self.__set_perturbed_injection(bus_name, phases, kw_inj, kvar_inj)
        self.dss.Text.Command("solve")
        if self.per_unit:
            return self.get_node_voltages_mag_pu()
        else:
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
