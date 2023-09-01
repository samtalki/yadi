"""
OpenDSS Quasi-Static Time Series Simulation Data Structure
@author: Samuel Talkington
MIT License
October 6th, 2021

"""

import numpy as np
import pandas as pd
import yadi.dss.model as model 
import warnings
from tqdm import tqdm
import pathlib
import os
from calendar import monthrange

#Optional: Turn off complex value warnings
#warnings.simplefilter("ignore", np.ComplexWarning)

class DSS_Timeseries(model.DSS_Data):

    def __init__(
            self,
            redirects,
            time_step,
            simulation_steps,
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
        super().__init__(redirects,precompile=precompile)
        self.time_step = time_step
        self.simulation_steps = simulation_steps
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
        self.initialize_qsts_duty()

        # All electircal nodes in the system - includes all individual conductors/phases
        nodes = self.dss.Circuit.YNodeOrder()
        n_nodes = len(nodes)
        
        if(self.nodal_mvts_dfs is None):
            #Get the timeseries arrays
            self.__run_qsts_duty(nodes,n_nodes)
        
        nodal_mvts_dfs = dict()
        dt_index = pd.date_range(start='1/1/2020',periods=self.simulation_steps,freq='1H')
        
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
        Run a "Quasi-Static Time-Series" and get multivariate timeseries dataets of voltage phasors, complex powers, and currents, for each node
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

        #Set internal fields for the data structure
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

    def get_monitor_timeseries(self,element_name,element_type="Load"):
        """
        Gets the voltage, active, and reactive power timeseries dictionary for a single elemented in the system: 
        D_i = {(V_i,t,P_i,t,Q_i,t)}_{t=1,..,M}

        Parameters:
        ---
            dss: the dss object
            element: name of the lement
        """
        #simulation_steps = int(60*60*24 / time_step) #Number of simulation steps in seconds
        #set_monitor(inj_node) #Setup the real and reactive power node monitor
        #initialize_qsts(dss,time_step,simulation_steps) #QSTS actions..
        #run_qsts_year(dss)
        
        voltage_ts = self.__export_monitor_voltage(element_name)
        power_ts = self.__export_monitor_power(element_name)    

        D = {
            'voltage_ts':voltage_ts,
            'power_ts':power_ts
        }
        return D


    def __export_monitor(self,monitor_name,verbose=False):
        """Exports a single monitor to a dataframe"""
        err = self.dss.run_command("export monitors {monitor_name}".format(
                monitor_name=monitor_name
            )
        )
        print('Monitor {monitor_name} Export Returned {err}'.format(
            monitor_name=monitor_name,
            err=err
        ))
        monitor_info = self.dss.utils.monitors_to_dataframe() #Get the monitor info df
        #Get the monitor info df index for monitor_name
        monitor_index = monitor_info.index.get_loc(monitor_info.index[monitor_info.index == monitor_name][0])
        
        #Make timeseries_df from exported csv
        timeseries_df = pd.read_csv(monitor_info['FileName'][monitor_index],sep=r'\s*,\s*',
                        header=0, encoding='ascii', engine='python')
        if(verbose):
            print('monitor_name is: ',monitor_name)
            print('monitor_idex is: ',monitor_index)
        return timeseries_df



    def set_loadshape(self,loadshape_path,loadshape_name='loadshape1'):
        """
        Sets a loadshape for all loads
        """
        self.dss.run_command(
            "Redirect {loadshape_path}".format(
                loadshape_path = loadshape_path
            )
        )
        self.dss.run_command(
            "batchedit load..*  yearly={loadshape_name} ".format(
                loadshape_name=loadshape_name
            )
            
        ) #change all loads            
    

    def __check_qsts_initialization(self):
        """Check if QSTS has been properly initialized"""
         #Check to see if QSTS is initialized
        if(not self.__qsts_initialized):
            warnings.warn("QSTS has not been initialized. Initiailizing before run.")
            self.compile_dss(self.redirects)
            self.initialize_qsts_duty()
        elif(self.__qsts_complete):
            warnings.warn("QSTS has already been run for the input files. Recompiling before run...")
            self.compile_dss(self.redirects)
            self.initialize_qsts_duty()

    #  #################################################
    #  ######### native Opendss - monthly QSTS #########
    #  #################################################

    def initialize_qsts_duty(self, monitor_loads=False, verbose=False):
        """
        Initialize a duty-mode Quasi-Static Time Series simulation.

        Params:
            monitor_loads (boolean): Whether or not to set monitors on all loads.

        """
        errs = []

        if(monitor_loads):
            self.__set_monitor_all_loads(verbose=verbose)
        errs.append(
            self.dss.run_command('Set controlmode=static')
        )
        errs.append(
            self.dss.run_command("Set mode=duty "
                                 f"number={self.simulation_steps} "
                                 f"stepsize={self.time_step}")
        )
        errs.append(
            self.dss.run_command('Set maxcontroliter=600')
        )
        print('QSTS Initialized, Returned: ', [err for err in errs])
        self.__qsts_initialized = True

    def run_monthly(self, scriptPath, month):
        """
        Compute monthly voltage, active, and reactive power timeseries dictionary for a single node i in the system:
        D_i = {(V_i,t,P_i,t,Q_i,t)}_{t=1,..,M} for all i

        Parameters:
        ---
            month: {01-jan, 02-feb, ....}
        """
        self.scriptPath = scriptPath
        self.monthlyDemand_dir = pathlib.Path(self.scriptPath).joinpath("test_cases", "secondary_test_network", "Profiles", "MonthlyDemand")
        if not os.path.isdir(self.monthlyDemand_dir):
            os.mkdir(self.monthlyDemand_dir)
        # load residential demand
        loadShapes = self.__load_LoadShapePerMonth(month)
        # set monthly loadShapes
        self.__setAllLoadShapes(loadShapes)
        # run routine with modified loadShapes
        self.__run_qsts_OpenDSS_duty()

    def __setAllLoadShapes(self, loadShapes):
        "Method to modify loadShapes from a DSS file"
        loadShapeNames = self.dss.LoadShape.AllNames()
        for n, loadShapeName in enumerate(loadShapeNames):
            if loadShapeName == 'default':
                continue
            # extract profiles
            Pmult = list(loadShapes.loc[:, loadShapeName + "_Pmult"].values)
            if loadShapeName + "_Qmult" in loadShapes.columns:
                Qmult = list(loadShapes.loc[:, loadShapeName + "_Qmult"].values)
            else:
                Qmult = None

            if Qmult is not None:
                self.__modifyLoadShapePQ(Pmult, Qmult, loadShapeName)
            else:
                self.__modifyLoadShapeP(Pmult, loadShapeName)

    def __modifyLoadShapeP(self, Pmult, name):
        self.dss.run_command(f"edit loadshape.{name} "
                             f"Npts={len(Pmult)} "
                             f"mult={Pmult} ")

    def __modifyLoadShapePQ(self, Pmult, Qmult, name):
        self.dss.run_command(f"edit loadshape.{name} "
                             f"Npts={len(Pmult)} "
                             f"mult={Pmult} "
                             f"Qmult={Qmult}")

    def __extract_loadShapes(self):
        """extract loadshapes from dss file"""
        loadShapeNames = self.dss.LoadShape.AllNames()
        loadShape_dict = dict()
        for n, loadShapeName in enumerate(loadShapeNames):
            if loadShapeName == 'default':
                continue
            # set active loadshape using its name
            self.dss.LoadShape.Name(loadShapeName)
            # checkName = self.dss.LoadShape.Name()
            # get Pmult and Qmult
            Pmult = self.dss.LoadShape.PMult()
            Qmult = self.dss.LoadShape.QMult()
            # Pmult_len = len(Pmult)
            if len(Pmult) != 1:
                loadShape_dict[loadShapeName + "_Pmult"] = np.asarray(Pmult)
            if len(Qmult) != 1:
                loadShape_dict[loadShapeName + "_Qmult"] = np.asarray(Qmult)
        loadShapes = pd.DataFrame().from_dict(loadShape_dict)
        return loadShapes

    def __split_loadShapes(self, loadShapes):
        """split loadshapes by months"""
        skipRows = 0
        monthsForIter = ["01", "02", "03", "04", "05", "06",
                         "07", "08", "09", "10", "11", "12"]
        for it, monthIter in enumerate(monthsForIter):
            daysInMonth = monthrange(2020, int(monthIter))
            hoursInMonth = 24 * daysInMonth[1]  # number of hours in month
            # define the name of the monthly demand file
            monthlyDemand_path = pathlib.Path(self.monthlyDemand_dir).joinpath(f"month_{monthIter}_profile.pkl")
            if not os.path.isfile(monthlyDemand_path):
                dfDemand = loadShapes.iloc[skipRows:hoursInMonth, :]
                # create index date range
                if monthIter == "12":
                    time = pd.date_range(start=f"2020-{monthIter}-01", end="2021-01-01", freq="H")
                else:
                    time = pd.date_range(start=f"2020-{monthIter}-01", end=f"2020-{monthsForIter[it + 1]}-01", freq="H")
                # transform string index into datetime index
                dfDemand.index = pd.to_datetime(time[:-1])
                # call method for processing series
                dfDemand.to_pickle(monthlyDemand_path)
            else:
                skipRows += hoursInMonth

    def __load_LoadShapePerMonth(self, month):
        """load demand per month"""
        # extract demand
        monthlyDemand_path = pathlib.Path(self.monthlyDemand_dir).joinpath(f"month_{month}_profile.pkl")
        if not os.path.isfile(monthlyDemand_path):
            self.__split_loadShapes(self.__extract_loadShapes())
        dfDemand = pd.read_pickle(monthlyDemand_path)
        self.simulation_steps = len(dfDemand)
        self.demand = dfDemand
        return dfDemand

    def __set_monitor(self, element_name, element_type, mon_name_prefix, power=True, voltage=True, verbose=False):
        """Sets a monitor on element_name of element_type"""
        if(power):  # If power monitor is enabled
            mon_name = mon_name_prefix + "power"
            err1 = self.dss.run_command(f"New Monitor.{mon_name} "
                                        f"Element={element_type}.{element_name} "
                                        f"terminal=1 PPolar=no mode=1")
            if(verbose):
                print("Monitor type: ", mon_name_prefix + "power", " placed on", element_type, " name: ", element_name, "with errors: ", err1)

        if(voltage):
            mon_name = mon_name_prefix + "voltage"
            err2 = self.dss.run_command(f"New Monitor.{mon_name} "
                                        f"Element={element_type}.{element_name} "
                                        f"terminal=1 vipolar=yes mode=0")
            if(verbose):
                print("Monitor type: ", mon_name_prefix + "voltage", " placed on", element_type, " name: ", element_name, "with errors: ", err2)

    def __run_qsts_OpenDSS_duty(self):
        """
        Run a "Quasi-Static Time-Series" and get multivariate timeseries dataets of voltage magnitudes, complex powers, and currents, for each node
        """
        # Check QSTS initialization
        self.__check_qsts_initialization()
        # Run Duty mode qsts
        # whole duty
        self.dss.run_command('solve')
        voltage_profiles, kw_profiles, kvar_profiles = self.__get_monitor_all_loads()
        self.__qsts_complete = True
        self.loadVolts = voltage_profiles
        self.loadKws = kw_profiles
        self.loadKvars = kvar_profiles

    def __set_monitor_all_loads(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        loads = self.dss.Loads.AllNames()
        for n, load_name in enumerate(loads):
            mon_name_prefix = "mon_" + str(load_name) + "_"
            self.__set_monitor(element_name=load_name, element_type="Load",
                               mon_name_prefix=mon_name_prefix, power=True,
                               voltage=True, verbose=verbose)

    def __get_monitor_all_loads(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        loads = self.dss.Loads.AllNames()
        voltage_dict = dict()
        kw_dict = dict()
        kvar_dict = dict()
        for n, load_name in enumerate(loads):
            volts, kws, kvars = self.__get_monitor_timeseries(element_name=load_name)
            voltage_dict[load_name + "voltage"] = volts
            kw_dict[load_name] = kws
            kvar_dict[load_name] = kvars
        voltage_profiles = pd.DataFrame.from_dict(voltage_dict)
        voltage_profiles = voltage_profiles.set_index(self.demand.index)
        kw_profiles = pd.DataFrame.from_dict(kw_dict)
        kw_profiles = kw_profiles.set_index(self.demand.index)
        kvar_profiles = pd.DataFrame.from_dict(kvar_dict)
        kvar_profiles = kvar_profiles.set_index(self.demand.index)
        return voltage_profiles, kw_profiles, kvar_profiles

    def __get_monitor_timeseries(self, element_name, element_type="Load"):
        """
        Gets the voltage, active, and reactive power timeseries dictionary for a single elemented in the system:
        {(V_i,t,P_i,t,Q_i)_t}_{t=1,..,M}

        Parameters:
        ---
            dss: the dss object
            element: name of the lement
        """
        voltage_ts = self.__export_monitor_voltage(element_name)
        kw_ts, kvar_ts = self.__export_monitor_power(element_name)
        return voltage_ts, kw_ts, kvar_ts

    def __export_monitor_voltage(self, element_name):
        """Gets the voltage timeseries for an element that is monitored"""
        monitor_name = "mon_" + element_name + "_voltage"
        # set the active monitor according to name
        self.dss.Monitors.Name(monitor_name)
        voltage_matrix = self.dss.Monitors.AsMatrix()  # N timesteps x M chanels (t, 0, v1, angle 1, ... I1, angle1, ...)
        return voltage_matrix[:, 2]  # interested in v1

    def __export_monitor_power(self, element_name):
        """Gets the active and reactive power timeseries for an element monitored"""
        monitor_name = "mon_" + element_name + "_power"
        # set the active monitor according to name
        self.dss.Monitors.Name(monitor_name)
        power_matrix = self.dss.Monitors.AsMatrix()  # N timesteps x M chanels (t, 0, P1, Q1, ....)
        return power_matrix[:, 2], power_matrix[:, 3]
