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

    def initialize_qsts_duty(self,monitor_loads=True,verbose=True):
        """
        Initialize a duty-mode Quasi-Static Time Series simulation.

        Params:
            monitor_loads (boolean): Whether or not to set monitors on all loads.  

        """
        errs = []

        if(monitor_loads):
            self.__set_monitor_all_loads(verbose=verbose)

        errs.append(
            self.dss.run_command(
                "Set mode=duty number=1 hour=0 sec=0 stepsize={time_step} sec=0".format(
                time_step=self.time_step
                )
            )
        )
        errs.append(
            self.dss.run_command('Set controlmode=static')
        )
        errs.append(
            self.dss.run_command('Set maxcontroliter=600')
        )
        print('QSTS Initialized, Returned: ',[err for err in errs])
        self.__qsts_initialized=True


    
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
        dt_index = pd.date_range(start='1/1/2020',periods=self.simulation_steps,freq='1H')
        
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


    def get_voltages_static(self):
        """Gets the static voltages for all buses"""
        err = self.dss.run_command('solve')
        if(not err==""):
            print(err)
        voltages = self.dss.Circuit.AllBusMagPu()
        return voltages,err

    def __export_monitor_voltage(self,element_name):
        """Gets the voltage timeseries for an element that is monitored"""
        monitor_name = "mon_"+element_name+"_voltage"
        voltage_ts = self.__export_monitor(self.dss,monitor_name)
        return voltage_ts

    def __export_monitor_power(self,element_name):
        """Gets the active and reactive power timeseries for an element monitored"""
        monitor_name = "mon_"+element_name+"_power"
        power_ts = self.__export_monitor(self.dss,monitor_name)
        return power_ts

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


    def __set_monitor_all_loads(self,verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        loads = self.dss.Loads.AllNames()
        for n,load_name in enumerate(loads):
            mon_name_prefix = "mon_"+str(load_name)+"_"
            self.__set_monitor(element_name=load_name,element_type="Load",
            mon_name_prefix=mon_name_prefix,power=True,voltage=True,verbose=verbose)


    def __set_monitor(self,element_name,element_type,mon_name_prefix,power=True,voltage=True,verbose=False):
        """Sets a monitor on element_name of element_type"""
        if(power): #If power monitor is enabled
            err1 = self.dss.run_command(
                "New Monitor.{mon_name} Element={element_type}.{element_name} mode=1, terminal=1, PPolar=no".format(
                mon_name=mon_name_prefix+"power",
                element_type=element_type,
                element_name=element_name
                )
            )
            if(verbose):
                print("Monitor type: ", mon_name_prefix+"power"," placed on", element_type," name: ",element_name,"with errors: ",err1)
        if(voltage):
            err2 = self.dss.run_command(
                "New Monitor.{mon_name} Element={element_type}.{element_name} mode=0, terminal=1".format(
                    mon_name=mon_name_prefix+"voltage",
                    element_type=element_type,
                    element_name=element_name
                )
            )
            if(verbose):
                print("Monitor type: ", mon_name_prefix+"voltage"," placed on", element_type," name: ",element_name,"with errors: ",err2)

    
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