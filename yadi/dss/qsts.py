"""
OpenDSS Quasi-Static Time Series Simulation Data Structure
@author: Samuel Talkington
MIT License
October 6th, 2021

"""

from tkinter import W
import numpy as np
import pandas as pd
import yadi.yadi.dss.model as model 
import warnings
from tqdm import tqdm
import pathlib
import os
from calendar import monthrange

#Optional: Turn off complex value warnings
#warnings.simplefilter("ignore", np.ComplexWarning)

class DSS_Timeseries(model.DSS_Data):

    def __init__(self,redirects,time_step,simulation_steps):
        super().__init__(redirects)
        self.time_step = time_step
        self.simulation_steps = simulation_steps

        #Initialize numpy MVTS arrays
        self.voltages_mvts = None
        self.currents_mvts = None
        self.complex_powers_mvts = None

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
        """
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


    def __run_qsts_duty(self,nodes,n_nodes):
        """
        Run a "Quasi-Static Time-Series" and get multivariate timeseries dataets of voltage magnitudes, complex powers, and currents, for each node
        """
        #Check QSTS initialization
        self.__check_qsts_initialization()
     
        #Set internal fields for the data structure
        self.voltages_mvts = np.empty((self.simulation_steps,n_nodes),dtype=np.cdouble) #voltage multivariate timeseries array MxN
        self.complex_powers_mvts = np.empty((self.simulation_steps,n_nodes),dtype=np.cdouble) #voltage multivariate timeseries array MxN
        self.currents_mvts = np.empty((self.simulation_steps,n_nodes),dtype=np.cdouble) #current multivariate timeseries array MxN
        
        #Run Duty mode qsts
        for it in tqdm (range (self.simulation_steps), desc="QSTS running..."):  
            err = self.dss.run_command('solve')
            if(err != ''):
                warnings.warn('OpenDSS Raised a QSTS error: ',err)     
            voltages_dict_t = self.get_node_voltages()
            currents_dict_t = self.get_node_currents() #get static current dict at timestep t
            complex_powers_dict_t = self.get_node_complex_powers() #get static voltage dict at timestep t
            for node_idx,node in enumerate(nodes):
                self.voltages_mvts[it,node_idx] = voltages_dict_t[node] #fill in the MVTS array
                self.currents_mvts[it,node_idx] = currents_dict_t[node] #fill in the MVTS array
                self.complex_powers_mvts[it,node_idx] = complex_powers_dict_t[node] #fill in the MVTS array
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


    def get_voltages_static(self):
        """Gets the static voltages for all buses"""
        err = self.dss.run_command('solve')
        if(not err==""):
            print(err)
        voltages = self.dss.Circuit.AllBusMagPu()
        return voltages,err

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
            self.initialize_qsts_duty()
        elif(self.__qsts_complete):
            warnings.warn("QSTS has already been run for the input files. Recompiling before run...")
            self.compile_dss(self.redirects)
            self.initialize_qsts_duty()

    #  #################################################
    #  ######### native Opendss - monthly QSTS #########
    #  #################################################

    def get_loadBusAndVln(self):
        "Method to extract load buses from a feeder"
        loadBusDict = dict()
        loadVlnDict = dict()
        elems = self.dss.Circuit.AllElementNames()
        for elem in elems:
            self.dss.Circuit.SetActiveElement(elem)
            if "Load" in elem:
                # extract load name
                loadName = elem.split(".")[1]
                # get bus name
                buses = self.dss.CktElement.BusNames()
                bus = buses[0]
                # save name
                loadBusDict[loadName] = bus
                # extract load line-to-neutral voltage
                self.dss.Loads.Name(loadName)
                loadVlnDict[loadName] = self.dss.Loads.kV()
        return pd.Series(loadBusDict), pd.Series(loadVlnDict)

    def get_lineEAmps(self):
        "Method to extract line emergency amps"
        lines = self.dss.Lines.AllNames()
        thermalLimitDict = dict()
        for line in lines:
            self.dss.Lines.Name(line)
            thermalLimitDict[line] = self.dss.Lines.NormAmps()
        return pd.Series(thermalLimitDict)

    def initialize_qsts_duty(self, monitor_trafos=True, monitor_lines=True, monitor_loads=True, verbose=False):
        """
        Initialize a duty-mode Quasi-Static Time Series simulation.

        Params:
            monitor_loads (boolean): Whether or not to set monitors on all loads.

        """
        errs = []

        if(monitor_loads):
            self.__set_monitor_all_loads(verbose=verbose)

        if(monitor_lines):
            self.__set_monitor_all_lines(verbose=verbose)

        if(monitor_lines):
            self.__set_monitor_all_trafos(verbose=verbose)

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

    def run_yearly(self, userDemand=None):
        """
        Compute monthly voltage, active, and reactive power timeseries dictionary for a single node i in the system:
        D_i = {(V_i,t,P_i,t,Q_i,t)}_{t=1,..,M} for all i

        Parameters:
        ---
            month: {01-jan, 02-feb, ....}
        """
        if userDemand is not None:
            self.__setAllLoadShapes(userDemand[0], userDemand[1])
            self.offset = 0
        else:
            self.offset = 3  # the dss duty lenght is shifted!
        # run routine with modified loadShapes
        self.__run_qsts_OpenDSS_duty()

    def run_monthly(self, scriptPath, month):
        """
        Compute monthly voltage, active, and reactive power timeseries dictionary for a single node i in the system:
        D_i = {(V_i,t,P_i,t,Q_i,t)}_{t=1,..,M} for all i

        Parameters:
        ---
            month: {01-jan, 02-feb, ....}
        """
        self.daysInThisMonth = monthrange(2019, int(month))
        self.scriptPath = scriptPath
        self.monthlyDemand_dir = pathlib.Path(self.scriptPath).joinpath("outputs", "monthlyDemand")
        if not os.path.isdir(self.monthlyDemand_dir):
            os.mkdir(self.monthlyDemand_dir)
        # load residential demand
        kwLoadShapes, kvarLoadShapes = self.__load_LoadShapePerMonth(month)
        # set monthly loadShapes
        self.__setAllLoadShapes(kwLoadShapes, kvarLoadShapes)
        # run routine with modified loadShapes
        self.__run_qsts_OpenDSS_duty()

    def __setAllLoadShapes(self, kwLoadShapes, kvarLoadShapes):
        "Method to modify loadShapes from a DSS file"
        loadShapeNames = self.dss.LoadShape.AllNames()
        for n, loadShapeName in enumerate(loadShapeNames):
            if loadShapeName == 'default':
                continue
            # extract profiles
            if loadShapeName in kwLoadShapes.columns:
                Pmult = list(kwLoadShapes.loc[:, loadShapeName].values)
            else:
                Pmult = None
            if loadShapeName in kvarLoadShapes.columns:
                Qmult = list(kvarLoadShapes.loc[:, loadShapeName].values)
            else:
                Qmult = None

            if (Qmult is not None) and (Pmult is not None):
                self.__modifyLoadShapePQ(Pmult, Qmult, loadShapeName)
            elif kwLoadShapes.shape[1] == kvarLoadShapes.shape[1]:
                self.dss.LoadShape.Name(loadShapeName)
                offset = 3
                Pmult = self.dss.LoadShape.PMult()
                self.__modifyLoadShapeP(Pmult[offset:], loadShapeName)
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
        kwLoadShape_dict = dict()
        kvarLoadShape_dict = dict()
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
                kwLoadShape_dict[loadShapeName] = np.asarray(Pmult)
            if len(Qmult) != 1:
                kvarLoadShape_dict[loadShapeName] = np.asarray(Qmult)
        kwLoadShapes = pd.DataFrame().from_dict(kwLoadShape_dict)
        kvarLoadShapes = pd.DataFrame().from_dict(kvarLoadShape_dict)
        return (kwLoadShapes, kvarLoadShapes)

    def __split_loadShapes(self, loadShapes):
        """split loadshapes by months"""
        kwLoadShapes = loadShapes[0]
        kvarLoadShapes = loadShapes[1]
        skipRows = 0
        monthsForIter = ["01", "02", "03", "04", "05", "06",
                         "07", "08", "09", "10", "11", "12"]
        for it, monthIter in enumerate(monthsForIter):
            daysInMonth = monthrange(2019, int(monthIter))
            hoursInMonth = 24 * daysInMonth[1]  # number of hours in month
            # define the name of the monthly demand file
            kwDemand_path = pathlib.Path(self.monthlyDemand_dir).joinpath(f"month_{monthIter}_kwProfile.pkl")
            kvarDemand_path = pathlib.Path(self.monthlyDemand_dir).joinpath(f"month_{monthIter}_kvarProfile.pkl")
            if not os.path.isfile(kwDemand_path):
                if monthIter == "12":
                    kwDemand = kwLoadShapes.iloc[skipRows:, :]
                    kvarDemand = kvarLoadShapes.iloc[skipRows:, :]
                    # create index date range
                    time = pd.date_range(start=f"2019-{monthIter}-01", end="2020-01-01", freq="H")
                else:
                    kwDemand = kwLoadShapes.iloc[skipRows:skipRows + hoursInMonth, :]
                    kvarDemand = kvarLoadShapes.iloc[skipRows:skipRows + hoursInMonth, :]
                    # create index date range
                    time = pd.date_range(start=f"2019-{monthIter}-01", end=f"2019-{monthsForIter[it + 1]}-01", freq="H")
                # transform string index into datetime index
                kwDemand.index = pd.to_datetime(time[:len(kwDemand)])
                kvarDemand.index = pd.to_datetime(time[:len(kvarDemand)])
                # call method for processing series
                kwDemand.to_pickle(kwDemand_path)
                kvarDemand.to_pickle(kvarDemand_path)
                skipRows += hoursInMonth
            else:
                skipRows += hoursInMonth

    def __load_LoadShapePerMonth(self, month):
        """load demand per month"""
        # extract demand
        kwDemand_path = pathlib.Path(self.monthlyDemand_dir).joinpath(f"month_{month}_kwProfile.pkl")
        kvarDemand_path = pathlib.Path(self.monthlyDemand_dir).joinpath(f"month_{month}_kvarProfile.pkl")
        if not os.path.isfile(kwDemand_path):
            self.__split_loadShapes(self.__extract_loadShapes())
        kwDemand = pd.read_pickle(kwDemand_path)
        kvarDemand = pd.read_pickle(kvarDemand_path)
        self.simulation_steps = len(kwDemand)
        self.kwDemand = kwDemand
        self.kvarDemand = kvarDemand
        return kwDemand, kvarDemand

    def __set_monitor(self, element_name, element_type, mon_name_prefix, power=True, voltage=True, verbose=False):
        """Sets a monitor on element_name of element_type"""
        if(power):  # If power monitor is enabled
            mon_name = mon_name_prefix + "power"
            err1 = self.dss.run_command(f"New Monitor.{mon_name} "
                                        f"Element={element_type}.{element_name} "
                                        "terminal=1 PPolar=no mode=1")
            if(verbose):
                print("Monitor type: ", mon_name_prefix + "power", " placed on", element_type, " name: ", element_name, "with errors: ", err1)

        if(voltage):
            mon_name = mon_name_prefix + "voltage"
            err2 = self.dss.run_command(f"New Monitor.{mon_name} "
                                        f"Element={element_type}.{element_name} "
                                        "terminal=1 vipolar=yes mode=0")
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
        lineIjk, linePjk, lineQjk = self.__get_monitor_all_lines()
        trafoIjk, trafoPjk, trafoQjk = self.__get_monitor_all_trafos()
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

    def __set_monitor_all_lines(self, verbose=False):
        """Sets timeseries power monitors on all lines before solving"""
        lines = self.dss.Lines.AllNames()
        for n, line_name in enumerate(lines):
            mon_name_prefix = "mon_" + str(line_name) + "_"
            self.__set_monitor(element_name=line_name, element_type="Line",
                               mon_name_prefix=mon_name_prefix, power=True,
                               voltage=True, verbose=verbose)

    def __set_monitor_all_trafos(self, verbose=False):
        """Sets timeseries power monitors on all trafos before solving"""
        trafos = self.dss.Transformers.AllNames()
        for n, trafo_name in enumerate(trafos):
            mon_name_prefix = "mon_" + str(trafo_name) + "_"
            self.__set_monitor(element_name=trafo_name, element_type="Transformer",
                               mon_name_prefix=mon_name_prefix, power=True,
                               voltage=True, verbose=verbose)

    def __set_monitor_all_loads(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        loads = self.dss.Loads.AllNames()
        for n, load_name in enumerate(loads):
            mon_name_prefix = "mon_" + str(load_name) + "_"
            self.__set_monitor(element_name=load_name, element_type="Load",
                               mon_name_prefix=mon_name_prefix, power=True,
                               voltage=True, verbose=verbose)

    def __get_monitor_all_trafos(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        trafos = self.dss.Transformers.AllNames()
        Ijk_dict = dict()
        Pjk_dict = dict()
        Qjk_dict = dict()
        for n, trafo_name in enumerate(trafos):
            Ijk, Pjk, Qjk = self.__get_monitor_timeseries(name=trafo_name, type="Transformer")
            if (len(Ijk.shape) > 1) and (len(Pjk.shape) > 1) and (len(Qjk.shape) > 1):
                self.dss.Transformers.Name(trafo_name)  # set current trafo as active
                phases = self.dss.CktElement.NumPhases()
                for ph in range(phases):
                    Ijk_dict[trafo_name + f".{ph + 1}"] = Ijk[:, ph]
                    Pjk_dict[trafo_name + f".{ph + 1}"] = Pjk[:, ph]
                    Qjk_dict[trafo_name + f".{ph + 1}"] = Qjk[:, ph]
            else:
                Ijk_dict[trafo_name] = Ijk
                Pjk_dict[trafo_name] = Pjk[:, 0]
                Qjk_dict[trafo_name] = Qjk[:, 0]
        
        Ijk_profiles = pd.DataFrame.from_dict(Ijk_dict)
        Ijk_profiles = Ijk_profiles[self.offset:]
        Pjk_profiles = pd.DataFrame.from_dict(Pjk_dict)
        Pjk_profiles = Pjk_profiles[self.offset:]
        Qjk_profiles = pd.DataFrame.from_dict(Qjk_dict)
        Qjk_profiles = Qjk_profiles[self.offset:]

        dt_index = pd.date_range(start='1/1/2019', periods=self.simulation_steps - self.offset, freq='H')
        Ijk_profiles = Ijk_profiles.set_index(dt_index)
        Pjk_profiles = Pjk_profiles.set_index(dt_index)
        Qjk_profiles = Qjk_profiles.set_index(dt_index)
        return Ijk_profiles, Pjk_profiles, Qjk_profiles

    def __get_monitor_all_lines(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        lines = self.dss.Lines.AllNames()
        Ijk_dict = dict()
        Pjk_dict = dict()
        Qjk_dict = dict()
        for n, line_name in enumerate(lines):
            Ijk, Pjk, Qjk = self.__get_monitor_timeseries(name=line_name, type="Line")
            if (len(Ijk.shape) > 1) and (len(Pjk.shape) > 1) and (len(Qjk.shape) > 1):
                self.dss.Lines.Name(line_name)  # set current line as active
                phases = self.dss.Lines.Phases()
                for ph in range(phases):
                    Ijk_dict[line_name + f".{ph + 1}"] = Ijk[:, ph]
                    Pjk_dict[line_name + f".{ph + 1}"] = Pjk[:, ph]
                    Qjk_dict[line_name + f".{ph + 1}"] = Qjk[:, ph]
            else:
                Ijk_dict[line_name] = Ijk
                Pjk_dict[line_name] = Pjk[:, 0]
                Qjk_dict[line_name] = Qjk[:, 0]
        
        Ijk_profiles = pd.DataFrame.from_dict(Ijk_dict)
        Ijk_profiles = Ijk_profiles[self.offset:]
        Pjk_profiles = pd.DataFrame.from_dict(Pjk_dict)
        Pjk_profiles = Pjk_profiles[self.offset:]
        Qjk_profiles = pd.DataFrame.from_dict(Qjk_dict)
        Qjk_profiles = Qjk_profiles[self.offset:]

        dt_index = pd.date_range(start='1/1/2019', periods=self.simulation_steps - self.offset, freq='H')
        Ijk_profiles = Ijk_profiles.set_index(dt_index)
        Pjk_profiles = Pjk_profiles.set_index(dt_index)
        Qjk_profiles = Qjk_profiles.set_index(dt_index)
        return Ijk_profiles, Pjk_profiles, Qjk_profiles

    def __get_monitor_all_loads(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        loads = self.dss.Loads.AllNames()
        voltage_dict = dict()
        kw_dict = dict()
        kvar_dict = dict()
        for n, load_name in enumerate(loads):
            volts, kws, kvars = self.__get_monitor_timeseries(load_name)
            voltage_dict[load_name] = volts
            kw_dict[load_name] = kws
            kvar_dict[load_name] = kvars
        
        voltage_profiles = pd.DataFrame.from_dict(voltage_dict)
        voltage_profiles = voltage_profiles[self.offset:]
        kw_profiles = pd.DataFrame.from_dict(kw_dict)
        kw_profiles = kw_profiles[self.offset:]
        kvar_profiles = pd.DataFrame.from_dict(kvar_dict)
        kvar_profiles = kvar_profiles[self.offset:]

        dt_index = pd.date_range(start='1/1/2019', periods=self.simulation_steps - self.offset, freq='H')
        voltage_profiles = voltage_profiles.set_index(dt_index)
        kw_profiles = kw_profiles.set_index(dt_index)
        kvar_profiles = kvar_profiles.set_index(dt_index)
        return voltage_profiles, kw_profiles, kvar_profiles

    def __get_monitor_timeseries(self, name, type="Load"):
        """
        Gets the voltage, active, and reactive power timeseries dictionary for a single elemented in the system:
        {(V_i,t,P_i,t,Q_i)_t}_{t=1,..,M}

        Parameters:
        ---
            dss: the dss object
            element: name of the lement
        """
        if type == "Line" or type == "Transformer":
            Ijk_ts = self.__export_monitor_voltage(name, type)
            Pjk_ts, Qjk_ts = self.__export_monitor_power(name, type)
            return Ijk_ts, Pjk_ts, Qjk_ts
        else:
            voltage_ts = self.__export_monitor_voltage(name, type)
            kw_ts, kvar_ts = self.__export_monitor_power(name, type)
            return voltage_ts, kw_ts, kvar_ts

    def __export_monitor_voltage(self, name, type):
        """Gets the voltage timeseries for an element that is monitored"""
        monitor_name = "mon_" + name + "_voltage"
        # set the active monitor according to name
        self.dss.Monitors.Name(monitor_name)
        voltage_matrix = self.dss.Monitors.AsMatrix()  # N timesteps x M chanels (t, 0, v1, angle 1, ... I1, angle1, ...)
        if type == "Load":
            return voltage_matrix[:, 2]  # interested in v1 for loads
        elif type == "Line":
            self.dss.Lines.Name(name)  # set current line as active
            phases = self.dss.Lines.Phases()
            if phases == 1:
                numCols = voltage_matrix.shape[1]
                # print(f"phase:{phases}-cols:{numCols}")
                return voltage_matrix[:, 4]  # interested in current magnitudes for the lines
            if phases == 3:
                numCols = voltage_matrix.shape[1]
                # print(f"phase:{phases}-cols:{numCols}")
                return voltage_matrix[:, 8::2]  # interested in current magnitudes for the lines
        elif type == "Transformer":
            self.dss.Transformers.Name(name)  # set current line as active
            # self.dss.Circuit.SetActiveElement(type + "." + name)
            phases = self.dss.CktElement.NumPhases()
            if phases == 1:
                numCols = voltage_matrix.shape[1]
                # print(f"phase:{phases}-cols:{numCols}")
                return voltage_matrix[:, 6]  # interested in current magnitudes for the lines
            if phases == 3:
                numCols = voltage_matrix.shape[1]
                # print(f"phase:{phases}-cols:{numCols}")
                return voltage_matrix[:, 10::2]  # interested in current magnitudes for the lines
        

    def __export_monitor_power(self, name, type):
        """Gets the active and reactive power timeseries for an element monitored"""
        monitor_name = "mon_" + name + "_power"
        # set the active monitor according to name
        self.dss.Monitors.Name(monitor_name)
        power_matrix = self.dss.Monitors.AsMatrix()  # N timesteps x M chanels (t, 0, P1, Q1, ....)
        if type == "Load":
            return power_matrix[:, 2], power_matrix[:, 3]  # interesteed in P1, Q1 for loads
        elif type == "Line":
            numCols = power_matrix.shape[1]
            # print(f"cols:{numCols}")
            return power_matrix[:, 2::2], power_matrix[:, 3::2]   # interesteed in Pjks and Qjks for the phases
        elif type == "Transformer":
            numCols = power_matrix.shape[1]
            # print(f"cols:{numCols}")
            return power_matrix[:, 2::2], power_matrix[:, 3::2]   # interesteed in Pjks and Qjks for the phases
