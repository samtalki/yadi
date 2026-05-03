"""Quasi-static time series (QSTS) dataset orchestration on top of OpenDSS."""

import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import yadi.dss.voltage_source as voltage_source


class DSS_Timeseries(voltage_source.DSS_VoltageSource):
    """Run a QSTS simulation and collect per-node voltage, current, and power timeseries."""

    def __init__(
        self,
        redirects: str | list[str],
        time_step: float,
        simulation_steps: int,
        simulation_mode: str = "duty",
        simulation_controlmode: str = "static",
        maxcontroliter: int = 50,
        miniterations: int = 1,
        maxiterations: int = 25,
        solution_number: int = 1,
        data_structure: str = "matrix",
        flow_direction: str = "from",
        verbose: bool = True,
        per_unit: bool = True,
        precompile: bool = True,
    ) -> None:
        super().__init__(redirects, precompile=precompile)

        self.time_step = time_step
        self.simulation_steps = simulation_steps
        self.simulation_mode = simulation_mode
        self.simulation_controlmode = simulation_controlmode
        self.maxcontroliter = maxcontroliter
        self.miniterations = miniterations
        self.maxiterations = maxiterations
        self.solution_number = solution_number

        if data_structure != "matrix":
            raise ValueError(
                f"data_structure={data_structure!r} is not supported; only 'matrix' is implemented."
            )
        if flow_direction not in ("from", "to"):
            raise ValueError(f"flow_direction={flow_direction!r} must be 'from' or 'to'.")
        self.data_structure = data_structure
        self.flow_direction = flow_direction
        self.verbose = verbose
        # per_unit is accepted for API parity with DSS_Sensitivities; not yet used here.
        self.per_unit = per_unit

        self.voltages_mvts: np.ndarray | None = None
        self.vmags_pu_mvts: np.ndarray | None = None
        self.currents_mvts: np.ndarray | None = None
        self.complex_powers_mvts: np.ndarray | None = None
        self.line_currents_mvts: np.ndarray | None = None
        self.nodal_mvts_dfs: dict[str, pd.DataFrame] | None = None

        self.__qsts_initialized = False
        self.__qsts_complete = False

    def __check_qsts_initialization(self, native=False):
        """Recompile and re-initialize QSTS if needed; warn on reset to flag lost state."""
        if not self.__qsts_initialized:
            warnings.warn("QSTS has not been initialized; initializing before run.")
            self.compile_dss()
            self.initialize_qsts(native)
        elif self.__qsts_complete:
            warnings.warn("QSTS already ran for these files; recompiling will reset DSS state.")
            self.compile_dss()
            self.initialize_qsts(native)

    def initialize_qsts(self, native, verbose=False):
        """Initialize a QSTS simulation; in native mode, place monitors on loads/lines/xfmrs."""
        if native:
            number = self.simulation_steps
            self.set_monitor_all_loads(verbose=verbose)
            self.set_monitor_all_lines(verbose=verbose)
            self.set_monitor_all_trafos(verbose=verbose)
        else:
            number = self.solution_number

        self.dss.Text.Command(f"Set controlmode={self.simulation_controlmode}")
        self.dss.Text.Command(
            f"Set mode={self.simulation_mode} "
            f"number={number} "
            f"stepsize={self.time_step} "
            f"maxcontroliter={self.maxcontroliter} "
            f"maxiterations={self.maxiterations} "
            f"miniterations={self.miniterations} "
        )
        self.__qsts_initialized = True

    def run(self) -> None:
        """Run the QSTS simulation and populate the per-node timeseries dataframes."""
        self.compile_dss()
        self.initialize_qsts(native=False)

        nodes = self.dss.Circuit.YNodeOrder()
        n_nodes = len(nodes)

        if self.nodal_mvts_dfs is None:
            self.__run_qsts_duty(nodes, n_nodes)

        nodal_mvts_dfs: dict[str, pd.DataFrame] = {}
        dt_index = pd.date_range(start="1/1/2019", periods=self.simulation_steps, freq="1h")
        assert self.voltages_mvts is not None and self.complex_powers_mvts is not None

        for i, node in enumerate(nodes):
            D_i = pd.DataFrame(index=dt_index, columns=["netloadV", "netloadP", "netloadQ"])
            D_i["netloadV"] = self.voltages_mvts[:, i]
            D_i["netloadP"] = np.real(self.complex_powers_mvts[:, i])
            D_i["netloadQ"] = np.imag(self.complex_powers_mvts[:, i])
            nodal_mvts_dfs[node] = D_i
        self.nodal_mvts_dfs = nodal_mvts_dfs

    def __run_qsts_duty(self, nodes, n_nodes):
        """Run QSTS and populate per-node voltage, current, and complex-power MVTS arrays."""
        names_lines = self.dss.Lines.AllNames()
        data_lines = self.get_line_data()

        # n_cond_lines must stay aligned with the OpenDSS line iterator below.
        try:
            n_cond_lines = [data_lines[name]["NumConductors"] for name in names_lines]
        except KeyError as e:
            raise RuntimeError(
                f"Missing NumConductors for element {e.args[0]!r}; cannot run QSTS."
            ) from e

        tot_cond_lines = int(np.sum(n_cond_lines, dtype=int))

        self.voltages_mvts = np.empty((self.simulation_steps, n_nodes), dtype=np.cdouble)
        self.vmags_pu_mvts = np.empty((self.simulation_steps, n_nodes), dtype=np.double)
        self.complex_powers_mvts = np.empty((self.simulation_steps, n_nodes), dtype=np.cdouble)
        self.currents_mvts = np.empty((self.simulation_steps, n_nodes), dtype=np.cdouble)
        self.line_currents_mvts = np.empty(
            (self.simulation_steps, tot_cond_lines), dtype=np.cdouble
        )

        for it in tqdm(range(self.simulation_steps), desc="QSTS running..."):
            self.solve()
            voltages_dict_t = self.get_node_voltages()
            vmags_pu_dict_t = self.get_node_voltages_mag_pu()
            currents_dict_t = self.get_node_currents()
            complex_powers_dict_t = self.get_node_complex_powers()
            line_currents_dict_t = self.get_line_currents(structure=self.data_structure)

            for node_idx, node in enumerate(nodes):
                self.vmags_pu_mvts[it, node_idx] = vmags_pu_dict_t[node]
                self.voltages_mvts[it, node_idx] = voltages_dict_t[node]
                self.currents_mvts[it, node_idx] = currents_dict_t[node]
                self.complex_powers_mvts[it, node_idx] = complex_powers_dict_t[node]

            term_row = 0 if self.flow_direction == "from" else 1
            line_idx, cond_idx, line = 0, 0, self.dss.Lines.First()
            while line:
                name = self.dss.Lines.Name()
                n_cond = n_cond_lines[line_idx]
                self.line_currents_mvts[it, cond_idx : cond_idx + n_cond] = line_currents_dict_t[
                    name
                ][term_row, :n_cond]
                cond_idx += n_cond
                line_idx += 1
                line = self.dss.Lines.Next()

        self.__qsts_complete = True

    def get_node_qsts_df(self, node):
        """Return the per-node MVTS dataframe; requires `run()` to have been called first."""
        if self.nodal_mvts_dfs is None:
            raise RuntimeError("QSTS has not been run yet; call `.run()` before requesting frames.")
        return self.nodal_mvts_dfs[node]

    def get_system_deviations(self, granularity: float | None = None) -> dict[str, np.ndarray]:
        """Finite-difference V/P/Q across the QSTS dataset; defaults to `self.time_step` seconds."""
        if self.voltages_mvts is None or self.complex_powers_mvts is None:
            raise RuntimeError("QSTS has not been run yet; call `.run()` before deviations.")
        if granularity is None:
            granularity = self.time_step
        N_nodes = self.voltages_mvts.shape[1]
        T_steps = len(self.voltages_mvts) - 1
        assert (
            T_steps == len(self.complex_powers_mvts) - 1
            and N_nodes == self.complex_powers_mvts.shape[1]
        )

        deltaV = np.diff(np.abs(self.voltages_mvts), axis=0).T / granularity
        deltaP = np.diff(np.real(self.complex_powers_mvts), axis=0).T / granularity
        deltaQ = np.diff(np.imag(self.complex_powers_mvts), axis=0).T / granularity
        return {"deltaV": deltaV, "deltaP": deltaP, "deltaQ": deltaQ}

    def run_native_qsts(self, userDemand=None):
        """Run a native OpenDSS QSTS via Monitors; modifies loadshapes if `userDemand` is given."""
        if userDemand is not None:
            self.setAllLoadShapes(userDemand[0], userDemand[1])
        self.__run_native_qsts_duty()

    def __run_native_qsts_duty(self):
        self.__check_qsts_initialization(native=True)
        self.run_command("solve")

        voltage_profiles, kw_profiles, kvar_profiles = self.get_monitor_all_loads()
        lineIjk, linePjk, lineQjk = self.get_monitor_all_lines()
        trafoIjk, trafoPjk, trafoQjk = self.get_monitor_all_trafos()
        self.__qsts_complete = True

        self.loadVolts = voltage_profiles
        self.loadKws = kw_profiles
        self.loadKvars = kvar_profiles

        self.lineIjks = lineIjk
        self.linePjks = linePjk
        self.lineQjk = lineQjk

        self.trafoIjks = trafoIjk
        self.trafoPjks = trafoPjk
        self.trafoQjk = trafoQjk

    def run_PMD_qsts(self):
        """Run a Python-side QSTS that populates PowerModelsDistribution time-series structures."""
        self.__run_PMD_qsts_duty()

    def __run_PMD_qsts_duty(self):
        self.__check_qsts_initialization()
        self.create_buses()
        self.create_lines()
        self.create_xfmrs()
        self.create_loads()

        for _ in tqdm(range(self.simulation_steps), desc="QSTS running..."):
            self.run_command("solve")
            self.read_bus_voltages()
            self.read_line_power()
            self.read_xfmr_power()
            self.read_load_power()

        self.__qsts_complete = True
