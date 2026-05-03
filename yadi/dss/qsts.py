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
        """Construct a QSTS simulator. `data_structure` is "matrix" or "dict";
        `flow_direction` is "from" or "to". `per_unit` is currently unused."""
        super().__init__(redirects, precompile=precompile)

        self.time_step = time_step
        self.simulation_steps = simulation_steps
        self.simulation_mode = simulation_mode
        self.simulation_controlmode = simulation_controlmode
        self.maxcontroliter = maxcontroliter
        self.miniterations = miniterations
        self.maxiterations = maxiterations
        self.solution_number = solution_number

        self.data_structure = data_structure
        self.flow_direction = flow_direction
        self.verbose = verbose
        self.per_unit = per_unit

        # (m x n) multivariate timeseries arrays, populated by run().
        self.voltages_mvts: np.ndarray | None = None
        self.vmags_pu_mvts: np.ndarray | None = None
        self.currents_mvts: np.ndarray | None = None
        self.complex_powers_mvts: np.ndarray | None = None
        self.line_currents_mvts: np.ndarray | None = None
        self.xfmr_currents_mvts: np.ndarray | None = None
        self.nodal_mvts_dfs: dict[str, pd.DataFrame] | None = None

        self.__qsts_initialized = False
        self.__qsts_complete = False

    def __check_qsts_initialization(self, native=False):
        """Check if QSTS has been properly initialized"""
        # Check to see if QSTS is initialized
        if not self.__qsts_initialized:
            warnings.warn("QSTS has not been initialized. Initiailizing before run.")
            self.compile_dss()
            self.initialize_qsts(native)

        elif self.__qsts_complete:
            # warnings.warn("QSTS has already been run for the input files. Recompiling before run...")
            self.compile_dss()
            self.initialize_qsts(native)

    def initialize_qsts(self, native, verbose=False):
        """
        Initialize a chosen-mode Quasi-Static Time Series simulation.

        Params:
            monitor_loads (boolean): Whether or not to set monitors on all loads.

        """
        errs = []

        if native:
            number = self.simulation_steps

            self.set_monitor_all_loads(verbose=verbose)

            self.set_monitor_all_lines(verbose=verbose)

            self.set_monitor_all_trafos(verbose=verbose)
        else:
            number = self.solution_number

        errs.append(self.dss.Text.Command(f"Set controlmode={self.simulation_controlmode}"))
        errs.append(
            self.dss.Text.Command(
                f"Set mode={self.simulation_mode} "
                f"number={number} "
                f"stepsize={self.time_step} "
                f"maxcontroliter={self.maxcontroliter} "
                f"maxiterations={self.maxiterations} "
                f"miniterations={self.miniterations} "
            )
        )

        # print('QSTS Initialized, Returned: ', [err for err in errs])
        self.__qsts_initialized = True

    #  #################################################
    #  ######### run net node injection QSTS #########
    #  #################################################

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
        """
        Run a "Quasi-Static Time-Series" and get multivariate timeseries dataset of
        voltage phasors, complex powers, and currents, for each node
        """
        # Get the names of all lines and transformers,
        names_lines = self.dss.Lines.AllNames()
        names_xfmrs = self.dss.Transformers.AllNames()

        # Get the data for all lines and transformers (Num conductors, phases, etc.)
        data_lines = self.get_line_data()
        data_xfmrs = self.get_xfmr_data()

        # Get the total number of condutors for all lines and transformers
        n_cond_lines, n_cond_xfmrs = [], []
        for name in names_lines:
            try:
                n_cond_lines.append(data_lines[name]["NumConductors"])
            except KeyError:
                warnings.warn(
                    f"Line {name} has no NumConductors attribute, not recording its num_conductors"
                )
        for name in names_xfmrs:
            try:
                n_cond_xfmrs.append(data_xfmrs[name]["NumConductors"])
            except KeyError:
                warnings.warn(
                    f"Xfmr {name} has no NumConductors attribute, not recording its num_conductors"
                )

        tot_cond_lines = int(np.sum(n_cond_lines, dtype=int))
        tot_cond_xfmrs = int(np.sum(n_cond_xfmrs, dtype=int))

        # Set internal fields for the data structure
        self.voltages_mvts = np.empty(
            (self.simulation_steps, n_nodes), dtype=np.cdouble
        )  # voltage multivariate timeseries array MxN
        self.vmags_pu_mvts = np.empty(
            (self.simulation_steps, n_nodes), dtype=np.double
        )  # voltage magnitude multivariate timeseries array MxN
        self.complex_powers_mvts = np.empty(
            (self.simulation_steps, n_nodes), dtype=np.cdouble
        )  # voltage multivariate timeseries array MxN
        self.currents_mvts = np.empty(
            (self.simulation_steps, n_nodes), dtype=np.cdouble
        )  # current multivariate timeseries array MxN
        self.line_currents_mvts = np.empty(
            (self.simulation_steps, tot_cond_lines), dtype=np.cdouble
        )  # line current multivariate timeseries array MxN
        self.xfmr_currents_mvts = np.empty(
            (self.simulation_steps, tot_cond_xfmrs), dtype=np.cdouble
        )  # xfmr current multivariate timeseries array MxN

        # Run Duty mode qsts
        for it in tqdm(range(self.simulation_steps), desc="QSTS running..."):
            err = self.dss.Text.Command("solve")
            if err != "":
                warnings.warn("OpenDSS Raised a QSTS error: ", err)
            voltages_dict_t = self.get_node_voltages()
            vmags_pu_dict_t = self.get_node_voltages_mag_pu()
            currents_dict_t = self.get_node_currents()
            complex_powers_dict_t = self.get_node_complex_powers()
            line_currents_dict_t = self.get_line_currents(structure=self.data_structure)

            # Fill in the nodal bus injection arrays
            for node_idx, node in enumerate(nodes):
                self.vmags_pu_mvts[it, node_idx] = vmags_pu_dict_t[
                    node
                ]  # fill in the MVTS array of nodal voltage magnitudes in per unit
                self.voltages_mvts[it, node_idx] = voltages_dict_t[
                    node
                ]  # fill in the MVTS array of nodal voltage phasors
                self.currents_mvts[it, node_idx] = currents_dict_t[
                    node
                ]  # fill in the MVTS array of nodal currents
                self.complex_powers_mvts[it, node_idx] = complex_powers_dict_t[
                    node
                ]  # fill in the MVTS array of nodal complex power injections

            # Fill in the line flow arrays
            line_idx, line = 0, self.dss.Lines.First()
            while line:
                name = self.dss.Lines.Name()  ##NOTE deprecateD: #names_lines[line_idx]
                n_cond = n_cond_lines[line_idx]
                if self.data_structure == "dict":
                    warnings.warn(
                        "Line currents not yet supported for dict data structure, using matrix instead"
                    )
                    self.data_structure = "matrix"
                if self.data_structure == "matrix":
                    if self.flow_direction == "from":
                        self.line_currents_mvts[it, line_idx : line_idx + n_cond] = (
                            line_currents_dict_t[name][0, :]
                        )
                    elif self.flow_direction == "to":
                        self.line_currents_mvts[it, line_idx : line_idx + n_cond] = (
                            line_currents_dict_t[name][1, :]
                        )
                    else:
                        raise ValueError("Invalid flow direction. Options are 'from' or 'to'")
                else:
                    raise ValueError("Invalid data structure. Options are 'matrix' or 'dict'")
                line_idx += 1
                line = self.dss.Lines.Next()

            ### NOTE: Transformer QSTS is broken right now, need to fix
            # # Fill in the xfmr flow arrays
            # xfmr_idx,xfmr = 0,self.dss.Transformers.First()
            # while xfmr:
            #     name = self.dss.Transformers.Name() #NOTE: Deprecated method #names_xfmrs[xfmr_idx]
            #     n_cond = n_cond_xfmrs[xfmr_idx]
            #     if self.data_structure == 'dict':
            #         warnings.warn("Xfmr currents not yet supported for dict data structure, using matrix instead")
            #         self.data_structure = 'matrix'
            #     if self.data_structure == 'matrix':
            #         if self.flow_direction == 'from':
            #             self.xfmr_currents_mvts[it,xfmr_idx:xfmr_idx+n_cond] = xmfr_currents_dict_t[name][0,:]
            #         elif self.flow_direction == 'to':
            #             self.xfmr_currents_mvts[it,xfmr_idx:xfmr_idx+n_cond] = xmfr_currents_dict_t[name][1,:]
            #         else:
            #             raise ValueError("Invalid flow direction. Options are 'from' or 'to'")
            #     else:
            #         raise ValueError("Invalid data structure. Options are 'matrix' or 'dict'")
            #     xfmr_idx += 1
            #     xfmr = self.dss.Transformers.Next()

        self.__qsts_complete = True

    def get_node_qsts_df(self, node):
        """
        Gets the MVTS DF at a specific node
        """
        if self.nodal_mvts_dfs is None:
            warnings.warn("QSTS has not been run yet, running...")
            self.run()
            return self.nodal_mvts_dfs[node]
        else:
            return self.nodal_mvts_dfs[node]

    def get_system_deviations(self, granularity=900):
        """
        Construct multivariate timeseries datasets of finite differences of:
            - Voltage magnitudes,
            - Active powers,
            - Reactive powers
        For all buses in the system.
        Params:
            D_N (array-like): List or array of N timeseries dictionaries
            granularity (seconds): Timestep interval used for finite difference approximation of time derivatives

        """
        N_nodes = self.voltages_mvts.shape[1]  # total number of nodes
        T_steps = len(self.voltages_mvts) - 1  # total number of timesteps in the deviation vectors
        assert (
            T_steps == len(self.complex_powers_mvts) - 1
            and N_nodes == self.complex_powers_mvts.shape[1]
        )

        # preallocate
        deltaV = np.zeros((N_nodes, T_steps))
        deltaP = np.zeros((N_nodes, T_steps))
        deltaQ = np.zeros((N_nodes, T_steps))

        # find deviations
        for i, v_i in enumerate(np.abs(self.voltages_mvts).T):
            # voltage deviations
            deltaV[i, :] = np.diff(v_i) / granularity
        for i, s_i in enumerate(self.complex_powers_mvts.T):
            # active power
            p_i = np.real(s_i)
            # reactive power
            q_i = np.imag(s_i)
            # deviations
            deltaP[i, :] = np.diff(p_i) / granularity
            deltaQ[i, :] = np.diff(q_i) / granularity
        D_diff_N = {"deltaV": deltaV, "deltaP": deltaP, "deltaQ": deltaQ}
        return D_diff_N

    #  #################################################
    #  ######### run native OpenDSS QSTS #########
    #  #################################################

    def run_native_qsts(self, userDemand=None):
        """
        runs a native QSTS simulation from OpenDSS.

        Parameters:
        ---
            userDemand:
        """
        if userDemand is not None:
            self.setAllLoadShapes(userDemand[0], userDemand[1])

        # run routine with modified loadShapes
        self.__run_native_qsts_duty()

    def __run_native_qsts_duty(self):
        """
        Run a "Quasi-Static Time-Series" within OpenDSS and get multivariate
        timeseries a dataset of voltage magnitudes, complex powers, and currents based on Monitors.
        """
        # Check QSTS initialization
        self.__check_qsts_initialization(native=True)

        # Run Duty mode qsts
        self.run_command("solve")

        # get monitor information
        voltage_profiles, kw_profiles, kvar_profiles = self.get_monitor_all_loads()
        lineIjk, linePjk, lineQjk = self.get_monitor_all_lines()
        trafoIjk, trafoPjk, trafoQjk = self.get_monitor_all_trafos()
        self.__qsts_complete = True

        # load quantities
        self.loadVolts = voltage_profiles
        self.loadKws = kw_profiles
        self.loadKvars = kvar_profiles

        # line quantities
        self.lineIjks = lineIjk
        self.linePjks = linePjk
        self.lineQjk = lineQjk

        # trafo quantities
        self.trafoIjks = trafoIjk
        self.trafoPjks = trafoPjk
        self.trafoQjk = trafoQjk

    #  ##########################################################
    #  ######### run QSTS for Power Models Distribution #########
    #  ##########################################################

    def run_PMD_qsts(self):
        """
        This method runs a python-based QSTS simulation for populating PMD time_series dictionary.

        Parameters:
        ---
            dss: the dss object
            element: name of the lement

        """

        # run routine with
        self.__run_PMD_qsts_duty()

    def __run_PMD_qsts_duty(self):
        """
        Run a "Quasi-Static Time-Series" simulation and populate a PMD dictionary
        comprising time_series data for each structure, i.e., voltage magnitudes
        for buses and active powers for lines, transformers, and loads.
        """

        # Check QSTS initialization
        self.__check_qsts_initialization()

        # Create all structures
        self.create_buses()
        self.create_lines()
        self.create_xfmrs()
        self.create_loads()

        # Run Duty mode qsts
        for it in tqdm(range(self.simulation_steps), desc="QSTS running..."):
            # run routine one set at a time
            self.run_command("solve")

            # get electrical quantities at time t
            self.read_bus_voltages()
            self.read_line_power()
            self.read_xfmr_power()
            self.read_load_power()

        self.__qsts_complete = True
