"""
OpenDSS Quasi-Static Time Series Simulation Time Series Structure
@author: Samuel Talkington
MIT License
October 6th, 2021

"""

from tkinter import W
import numpy as np
import pandas as pd
import yadi.yadi.dss.voltage_source as voltage_source 
import warnings
from tqdm import tqdm
import pathlib
import os

#Optional: Turn off complex value warnings
#warnings.simplefilter("ignore", np.ComplexWarning)

class DSS_Timeseries(voltage_source.DSS_VoltageSource):

    def __init__(
            self,
            redirects,
            time_step,
            simulation_steps,
            simulation_mode='duty',
            simulation_controlmode='static',
            maxcontroliter=50,
            miniterations=1,
            maxiterations=25,
            solution_number=1, 
            data_structure='matrix', # 'matrix' or 'dict'
            flow_direction='from', # 'from' or 'to'
            verbose=True,
            per_unit=True,
            precompile=True):
        """
        Quasi-Static Time Series (QSTS) simulation data structure for OpenDSS networks.
        Inherits from the DSS_Data class.

        Parameters:
            - redirects (list): List of strings of filepaths to .dss files
            - time_step (int): Timestep in seconds
            - simulation_steps (int): Number of simulation steps
            - data_structure (str): Data structure to use for storing the timeseries data. Options are 'matrix' or 'dict'
            - flow_direction (str): The direction of the line and xfmr flow values. Options are 'from' or 'to'
            - verbose (boolean): whether or not to print verbose logs
            - per_unit (boolean): whether or not to return per unit values at all times (default is False) #TODO
        """
        super().__init__(redirects, precompile=precompile)

        #--- Simulation parameters ---#
        self.time_step = time_step
        self.simulation_steps = simulation_steps # Number of simulation steps (iterations) to run in the QSTS simulation
        self.simulation_mode=simulation_mode
        self.simulation_controlmode=simulation_controlmode
        self.maxcontroliter = maxcontroliter
        self.miniterations = miniterations
        self.maxiterations = maxiterations
        self.solution_number = solution_number #the number of solutions to find at each call of the solve command (iteraiton)

        #---- Data structure and flow direction ----#
        self.data_structure = data_structure
        self.flow_direction = flow_direction
        self.verbose = verbose
        self.per_unit = per_unit #TODO: Add optional functionality to force per unit in all data structures

        #Voltage MVTS arrays
        self.voltages_mvts = None # (m x n) multivariate timeseries of complex voltage phasors (not per unit)
        self.vmags_pu_mvts = None # (m x n) multivariate timesries of voltage magnitudes in per unit

        #Current and power MVTS arrays
        self.currents_mvts = None
        self.complex_powers_mvts = None

        #Line currents and xfmr currents multivariate timeseries (MVTS) arrays
        self.line_currents_mvts = None
        self.xfmr_currents_mvts = None

        #nodal mvts_dfs
        self.nodal_mvts_dfs = None

        #Inernal boolean to see if qsts has been initiliazed
        self.__qsts_initialized = False
        self.__qsts_complete = False
    
    def __check_qsts_initialization(self, native=False):
        """Check if QSTS has been properly initialized"""
         #Check to see if QSTS is initialized
        if(not self.__qsts_initialized):
            warnings.warn("QSTS has not been initialized. Initiailizing before run.")
            self.compile_dss()
            self.initialize_qsts(native)

        elif(self.__qsts_complete):
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

        if(native):
            number = self.simulation_steps

            self.set_monitor_all_loads(verbose=verbose)

            self.set_monitor_all_lines(verbose=verbose)

            self.set_monitor_all_trafos(verbose=verbose)
        else:
            number = self.solution_number

        errs.append(
            self.dss.run_command(f'Set controlmode={self.simulation_controlmode}')
        )
        errs.append(
            self.dss.run_command(f"Set mode={self.simulation_mode} "
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

    def run(self):
        """
        Compute the voltage, active, and reactive power timeseries dictionary for a single node i in the system: 
        D_i = {(V_i,t,P_i,t,Q_i,t)}_{t=1,..,M} for all i

        Parameters:
        ---
            dss: the dss object
            element: name of the lement


        TODO:
        ---ADD MORE CONTROL OVER THE MATRICES THAT ARE GENERATED --- ONLY VMAG PER UNIT.
        """
        #Check QSTS initialization
        #self.__check_qsts_initialization()
        self.compile_dss()
        self.initialize_qsts()

        # All electircal nodes in the system - includes all individual conductors/phases
        nodes = self.dss.Circuit.YNodeOrder()
        n_nodes = len(nodes)
        
        if(self.nodal_mvts_dfs is None):
            #Get the timeseries arrays
            self.__run_qsts_duty(nodes,n_nodes)
        
        nodal_mvts_dfs = dict()
        dt_index = pd.date_range(start='1/1/2019', periods=self.simulation_steps, freq='1H')
        
        for i,node in enumerate(nodes):
            D_i = pd.DataFrame(index = dt_index,columns=['netloadV','netloadP','netloadQ'])
            D_i['netloadV'] = self.voltages_mvts[:,i]
            D_i['netloadP'] = np.real(self.complex_powers_mvts[:,i])
            D_i['netloadQ'] = np.imag(self.complex_powers_mvts[:,i])
            nodal_mvts_dfs[node] = D_i          
        self.nodal_mvts_dfs = nodal_mvts_dfs

    #TODO: Check columns with zero voltages
    #TODO: Optional parameter to only save certain variables for memory efficiency
    #TODO: Enable no-neutral xfmr currents
    def __run_qsts_duty(self,nodes,n_nodes):
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
        n_cond_lines,n_cond_xfmrs = [],[]
        for name in names_lines:
            try:
                n_cond_lines.append(data_lines[name]['NumConductors'])
            except:
                #n_cond_lines.append(1)
                warnings.warn("Line {name} has no NumConductors attribute, not recording its num_conductors".format(name=name))
        for name in names_xfmrs:
            try:
                n_cond_xfmrs.append(data_xfmrs[name]['NumConductors'])
            except:
                #n_cond_xfmrs.append(1)
                warnings.warn("Xfmr {name} has no NumConductors attribute, not recording its num_conductors".format(name=name))

        tot_cond_lines = np.sum(n_cond_lines)
        tot_cond_xfmrs = np.sum(n_cond_xfmrs)

        # Set internal fields for the data structure
        self.voltages_mvts = np.empty((self.simulation_steps,n_nodes),dtype=np.cdouble) #voltage multivariate timeseries array MxN
        self.vmags_pu_mvts = np.empty((self.simulation_steps,n_nodes),dtype=np.double) #voltage magnitude multivariate timeseries array MxN
        self.complex_powers_mvts = np.empty((self.simulation_steps,n_nodes),dtype=np.cdouble) #voltage multivariate timeseries array MxN
        self.currents_mvts = np.empty((self.simulation_steps,n_nodes),dtype=np.cdouble) #current multivariate timeseries array MxN
        self.line_currents_mvts = np.empty((self.simulation_steps,tot_cond_lines),dtype=np.cdouble) #line current multivariate timeseries array MxN
        self.xfmr_currents_mvts = np.empty((self.simulation_steps,tot_cond_xfmrs),dtype=np.cdouble) #xfmr current multivariate timeseries array MxN

        #Run Duty mode qsts
        for it in tqdm (range (self.simulation_steps), desc="QSTS running..."):  
            err = self.dss.run_command('solve')
            if(err != ''):
                warnings.warn('OpenDSS Raised a QSTS error: ',err)     
            voltages_dict_t = self.get_node_voltages()
            vmags_pu_dict_t = self.get_node_voltages_mag_pu()
            currents_dict_t = self.get_node_currents() #get static current dict at timestep t
            complex_powers_dict_t = self.get_node_complex_powers() #get static voltage dict at timestep t
            line_currents_dict_t = self.get_line_currents(structure=self.data_structure)#,flow_direction=self.flow_direction # TODO: Add flow direction)
            # xmfr_currents_dict_t = self.get_xfmr_currents(structure=self.data_structure)#,flow_direction=self.flow_direction # TODO: Add flow direction)

            # Fill in the nodal bus injection arrays
            for node_idx,node in enumerate(nodes):
                self.vmags_pu_mvts[it,node_idx] = vmags_pu_dict_t[node] #fill in the MVTS array of nodal voltage magnitudes in per unit
                self.voltages_mvts[it,node_idx] = voltages_dict_t[node] #fill in the MVTS array of nodal voltage phasors
                self.currents_mvts[it,node_idx] = currents_dict_t[node] #fill in the MVTS array of nodal currents
                self.complex_powers_mvts[it,node_idx] = complex_powers_dict_t[node] #fill in the MVTS array of nodal complex power injections
            
            # Fill in the line flow arrays
            line_idx,line = 0,self.dss.Lines.First()
            while line:
                name = self.dss.Lines.Name() ##NOTE deprecateD: #names_lines[line_idx]
                n_cond = n_cond_lines[line_idx]
                if self.data_structure == 'dict':
                    warnings.warn("Line currents not yet supported for dict data structure, using matrix instead")
                    self.data_structure = 'matrix'
                if self.data_structure == 'matrix':
                    if self.flow_direction == 'from':
                        self.line_currents_mvts[it,line_idx:line_idx+n_cond] = line_currents_dict_t[name][0,:]
                    elif self.flow_direction == 'to':
                        self.line_currents_mvts[it,line_idx:line_idx+n_cond] = line_currents_dict_t[name][1,:]
                    else:
                        raise ValueError("Invalid flow direction. Options are 'from' or 'to'")
                else:
                    raise ValueError("Invalid data structure. Options are 'matrix' or 'dict'")
                line_idx+= 1
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
            
        self.__qsts_complete=True

    def get_node_qsts_df(self,node):
        """
        Gets the MVTS DF at a specific node
        """
        if(self.nodal_mvts_dfs is None):
            warnings.warn("QSTS has not been run yet, running...")
            self.run()
            return self.nodal_mvts_dfs[node]
        else:
            return self.nodal_mvts_dfs[node]

    def get_system_deviations(self,granularity=900):
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
        N_nodes = self.voltages_mvts.shape[1] #total number of nodes
        T_steps = len(self.voltages_mvts)-1 #total number of timesteps in the deviation vectors
        assert T_steps == len(self.complex_powers_mvts)-1 and N_nodes == self.complex_powers_mvts.shape[1]
        
        #preallocate
        deltaV = np.zeros((N_nodes,T_steps)) 
        deltaP = np.zeros((N_nodes,T_steps))
        deltaQ = np.zeros((N_nodes,T_steps))
        
        #find deviations
        for i,v_i in enumerate(np.abs(self.voltages_mvts).T):
            #voltage deviations
            deltaV[i,:] = np.diff(v_i)/granularity
        for i,s_i in enumerate(self.complex_powers_mvts.T):
            #active power
            p_i = np.real(s_i)
            #reactive power
            q_i = np.imag(s_i)
            #deviations
            deltaP[i,:] = np.diff(p_i)/granularity
            deltaQ[i,:] = np.diff(q_i)/granularity
        D_diff_N = {
            "deltaV":deltaV,
            "deltaP":deltaP,
            "deltaQ":deltaQ
        }
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
        self.run_command('solve')

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
        
        #Run Duty mode qsts
        for it in tqdm (range (self.simulation_steps), desc="QSTS running..."):  
            # run routine one set at a time
            self.run_command('solve')

            # get electrical quantities at time t
            self.read_bus_voltages()
            self.read_line_power()
            self.read_xfmr_power()
            self.read_load_power()


        self.__qsts_complete=True