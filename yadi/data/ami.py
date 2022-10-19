#!/usr/bin/env python
""" 
Data structure for handling AMI data.
"""
__author__ = "Samuel Talkington"
__contact__ = "talkington@pm.me"
__copyright__ = "Copyright (c) 2021-Present Samuel Talkington, All Rights Reserved."
__deprecated__ = False
__license__ = "MIT"
__status__ = "Production"
__version__ = "0.0.1"


import numpy as np
import numpy.ma as ma
import pandas as pd
import warnings
import h5py 
from tqdm import tqdm


class AMIData:
    """
    Handles tabular advanced metering infrastrcuture (AMI) data describes by 3 MxN matrices P, Q, and V, which are active power, reactive power, and voltage magnitudes, respectively.
    All matrices are of the form (M measurements, N nodes).
    """

    def __init__(self,nodes=list(range(1379)),time_steps=list(range(35040)),
    data_path=default_data_path) -> None: #,truth_path=truth_path,capability_path=capability_path) -> None:
        """Generate a model-free HC input dataset for a set of nodes.

        Args:
            nodes (list, optional): list of nodes under study, all if none. Defaults to None (full nodes).
            time_steps (list, optional): range of time steps to consider. Defaults to None (full time horizon).
            data_path (string, optional): path to data. defaults to ./hc_baseline.
        """

        #Setup input data
        self.data_path__ = data_path
        #self.capability_path__ = capability_path
        #self.ground_truth_path__ = truth_path
        self.raw_data__ = None
        self.raw_ground_truth_data__ = None

        #Raw data matrix fields
        self.data = None # Dictionary of P,Q,V
        self.hc_baseline = None #Dictionary of HC baseline results
        self.capab_baseline = None #Matrix of hosting capaibility baseline results

        #Nodal AMI dataframes
        self.nodal_ami_dfs = None
        self.daytime_mask = None #Save the daytime mask 

        #Node parameter fields
        self.nodes = nodes
        self.N_nodes = None
        self.feature_index = None #Node names 
        self.__setup_nodes(nodes)

        #Time parameter fields
        self.time_steps = time_steps
        self.T_steps = None
        self.datetime_index = None #dataframe datetime axis
        self.__setup_times(time_steps)


    def get_datasets(self,interpolate='linear',inplace=True):
        """
        Make _(M x N)_ Data Matrices where M is the number of samples and N is the number of studied nodes.
        """ 
        raw_data = self.__read_raw_mat()
        #capab_baseline = h5py.File(self.capability_path__,"r")
        self.feature_index = np.asarray(list(raw_data['loadNames'][self.nodes])).flatten()  #Setup study node names
        data = {}
        #a= pd.read_csv(r'/home/sam/github/MoHCa/mohca/data/new_hc_baseline_xfmer/downstream.csv')
        #groups = [np.asarray(x[0].split(','),int) for x in a.values]
        data['P'] = np.transpose(np.asarray(raw_data['AMI_PkW']))
        data['Q'] = np.transpose(np.asarray(raw_data['AMI_QkVAR']))
        data['V'] = np.transpose(np.asarray(raw_data['AMI_Vpu']))
        #data['groups'] = groups
        hc_baseline = {
            'all':raw_data['HC_Vconstrained'][:,0],
            'day':raw_data['HC_Vconstrained'][:,1],
            #'capab':np.transpose(np.asarray(raw_data['HCapabilityAll']))
        }
        if(inplace):
            self.data = data
            self.hc_baseline = hc_baseline
            #self.capab_baseline = capab_baseline
            if(interpolate is not None):
                warnings.warn("Interpolating in place with default method: "+interpolate)
                self.interpolate()
            return self.data
        else:
            return data
    
    def get_daytime_datasets(self,inplace=False):
        """Get the daytime datasets"""
        if(self.nodal_ami_dfs is None):
            df0 = self.get_nodal_ami_dfs()[0]
        else:
            df0 = self.nodal_ami_dfs[0]
        self.daytime_mask = (df0.index.hour >= 9) & (df0.index.hour <= 15)
        day_data = {}
        day_data['V'] = self.data['V'][self.daytime_mask,:]
        day_data['P'] = self.data['P'][self.daytime_mask,:]
        day_data['Q'] = self.data['Q'][self.daytime_mask,:]
        #day_data['capab'] = self.data['capab'][self.daytime_mask,:]
        return day_data

    def get_nodal_ami_dfs(self,inplace=False):
        """
        Make pandas dataframes of the form:
        nodal_ami_dfs = [D_1,D_2,...,D_n]  where D_i = {(V_i1,...,V_in,P_i1,...,P_in,Q_i1,...,Q_in)}_t=1^T.
        """  
        #Setup data
        nodal_ami_dfs = []
        for P_i,Q_i,V_i in zip(self.data['P'].T,self.data['Q'].T,self.data['V'].T):
            D_i = np.vstack([P_i,Q_i,V_i]).T
            nodal_ami_dfs.append(
                pd.DataFrame(
                    data=D_i,
                    index=self.datetime_index,
                    columns=['P','Q','V']
                )
            )
        if(inplace):
            self.nodal_ami_dfs = nodal_ami_dfs
        else:
            return nodal_ami_dfs

    def interpolate(self,replace_data_matrices=True,method='linear'):
        """
        Interpolate AMI data to remove missing data
        
        Parameters:
            replace_data_matrices (bool): Whether to replace the internal P,Q,V matrices (default True)
        """
        if(self.nodal_ami_dfs is None):
            nodal_ami_dfs = self.get_nodal_ami_dfs()
            
        for i,df in tqdm(enumerate(nodal_ami_dfs),desc='Interpolating'):
            nodal_ami_dfs[i] = df.interpolate(method=method).iloc[1:]
            
        if(replace_data_matrices and self.data is not None):
            
            #Truncate datasets by 1
            self.data['P'] = np.zeros((self.T_steps-1,self.N_nodes))
            self.data['Q'] = np.zeros((self.T_steps-1,self.N_nodes))
            self.data['V'] = np.zeros((self.T_steps-1,self.N_nodes)) 
            
            # Truncate datetime index by 1
            self.datetime_index = self.datetime_index[1:] 
            
            for i,df in enumerate(nodal_ami_dfs):
                self.data['P'][:,i] = df['P']
                self.data['Q'][:,i] = df['Q']
                self.data['V'][:,i] = df['V']
        return nodal_ami_dfs

    def differentiate(self,granularity=1):
        """
        Get linear sensitivity measurement deviations for all buses in the system, i.e.
        
        y_ and A(X) where y=A(X)+z
        
        Params:
            D_N (array-like): List or array of N timeseries dictionaries 
            granularity (seconds): Timestep interval used for finite difference approximation of time derivatives
        
        """
        #Get number of deviation steps and check sizes
        M_diff = self.data['V'].shape[0] #total number of timesteps in the differentiated vectors ###################################note new version using np.gradient and is thus the same
        assert M_diff == self.data['P'].shape[0]
        assert self.N_nodes == self.data['P'].shape[1]
        
        keys = ['DV','DP','DQ']
        matrices = [self.data['V'],self.data['P'],self.data['Q']]
        
        #preallocate
        DV = np.zeros((M_diff,self.N_nodes)) 
        DP = np.zeros((M_diff,self.N_nodes))
        DQ = np.zeros((M_diff,self.N_nodes))
            
        for i,(V_i_T,P_i_T,Q_i_T)in enumerate(zip(self.data['V'].T,self.data['P'].T,self.data['Q'].T)):
            #deviations
            DV[:,i] = np.gradient(V_i_T)/granularity
            DP[:,i] = np.gradient(P_i_T)/granularity
            DQ[:,i] = np.gradient(Q_i_T)/granularity

        D_diff_N = {
            "DV":DV,
            "DP":DP,
            "DQ":DQ
        }
        
        return D_diff_N

    @staticmethod 
    def unbias(M):
        """Simple unbiasing via mean centering"""
        return M - np.mean(M,axis=0)

    @staticmethod
    def static_differentiate(voltage_mvts,real_mvts,reactive_mvts,granularity=1,diff_method=np.gradient):
        """
        Static differentiate for given measurements. 
        
        y_ and A(X) where y=A(X)+z
        
        Params:
            D_N (array-like): List or array of N timeseries dictionaries 
            granularity (seconds): Timestep interval used for finite difference approximation of time derivatives
            diff_method (function): Function for differentiating the data.
        """

        M,N = voltage_mvts.shape
        assert M == real_mvts.shape[0]
        assert N == real_mvts.shape[1]

        #preallocate
        DV,DP,DQ = [],[],[]
        # DV = np.zeros((M,N)) 
        # DP = np.zeros((M,N))
        # DQ = np.zeros((M,N))
        
        for i,(V_i_T,P_i_T,Q_i_T)in enumerate(zip(voltage_mvts.T,real_mvts.T,reactive_mvts.T)):
            #deviations
            DV.append(diff_method(V_i_T)/granularity)
            DP.append(diff_method(P_i_T)/granularity)
            DQ.append(diff_method(Q_i_T)/granularity)

        D_diff_N = {
            "DV":np.asarray(DV).T,
            "DP":np.asarray(DP).T,
            "DQ":np.asarray(DQ).T
        }
        
        return D_diff_N

    def __read_raw_mat(self):
        """
        Read the raw AMI data in .mat format
        """
        raw_data = h5py.File(self.data_path__,"r")
        return raw_data

    #TODO
    def __read_raw_csv(self):
        """
        Read raw data in CSV format
        """
        pass
    
    def __setup_times(self,time_steps):
        #Setup time horizon:
        if time_steps is None: 
            self.time_steps = list(range(35040))
        else:
            self.time_steps = time_steps 
        self.T_steps = len(self.time_steps) #save T_steps
        self.datetime_index = pd.date_range("2020",freq='15min',periods=self.T_steps)

    def __setup_nodes(self,nodes):
        #Setup study node indeces
        if nodes is None: 
            self.nodes =list(range(1379))
        else:
            self.nodes = nodes
        self.N_nodes = len(self.nodes) #Save number of nodes
        
