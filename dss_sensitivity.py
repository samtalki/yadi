"""
OpenDSS Sensitivity-Based Linear Approximation Model
Perturb and observe to form a linear approximation of the power flow equations (first order taylor expansion)
@author: Samuel Talkington
MIT License
October 6th, 2021
"""

import numpy as np
import dss_model
import pandas as pd
import warnings

class OpenDSS_Sensitivities(dss_model.OpenDSS_Data):

    def __init__(self,redirects,verbose=True):
        super().__init__(redirects,verbose)
        self.spv = None #sensitivity of vmag wrt active power
        self.sqv = None #sensitivity of vmag wrt reactive power
        self.spth = None #sensitivity of angles wrt active power
        self.sqth = None #sensitivity of angles wrt reactive power

    def get_svp(self):
        """
        Get the vmag sensitivity matrix w.r.t. active power i.e. the block of dV_i/dP_k derivatives of the inverse power flow Jacobian
        """
        spv,nodes,vph_base = self.__calc_sens_mat(indp_var="vmag",dep_var="p")
        self.spv = {
            "matrix":spv,
            "nodes":nodes,
            "vph_base":vph_base
        }
        return self.spv

    def get_svq(self):
        """
        Get the vmag sensitivity matrix w.r.t. reactive power i.e. the block of dV_i/dQ_k derivatives of the inverse power flow Jacobian
        """
        sqv,nodes,vph_base = self.__calc_sens_mat(indp_var="vmag",dep_var="q")
        self.sqv = {
            "matrix":sqv,
            "nodes":nodes,
            "vph_base":vph_base
        }
        return self.sqv

    def __calc_bus_sens_mat(self,indp_var='vmag',dep_var='p'):
        pass


    def __calc_sens_mat(self,indp_var='vmag',dep_var='p'):
        """
        Interface for calculating a sensitivity matrix

        Keyword arguments:
        indp_var -- independent variable -- "vph" or "vmag"
        dep_var -- dependent variable -- "p" or "q".
        circuit_name -- the "short name" for the circuit e.g. "ieee13"
        """
        # os.chdir(self.dssDir)
        # self.dss.run_command('Compile "'+self.dssFile+'"')
        # self.dss.run_command('Set Controlmode = STATIC')
        
        #Precompile the DSS file
        self.compile_dss(self.redirects)
        self.dss.run_command('solve')
        
        #Get the ybus ordered nodes
        nodes = self.dss.Circuit.YNodeOrder()
        #print("nodes for the circuit: ",nodes)
        
        #Get the ybus-ordered array of phase voltages (basecase)
        vph_base = self.get_node_voltages()
        vph_base_yorder = [vph_base[i] for i in nodes]
        #print('Voltages basecase: ',vph_base)   
        
        #Iteratively solve the injections and retrieve the voltages 
        S_matrix = np.zeros((len(nodes),len(nodes)))
        
        for col_idx, node in enumerate(nodes):
            if(self.verbose):
                print("Node Perturbed: ",node)
            
            if(dep_var=="P" or dep_var=='p'):
                #Get a dictionary of voltages after active power perturbation
                vph_perturbed = self.__get_perturbed_nodal_voltages(bus_name=node,phases=1,kw_inj=-100,kvar_inj=0)
                #Get the Ybus-ordered array of pertubred voltages
                vph_perturbed_yorder = [vph_perturbed[i] for i in nodes] 
            elif(dep_var=="Q" or dep_var=='q'):
                #Get a dictionary of voltages after reactive power perturbation
                vph_perturbed = self.__get_perturbed_nodal_voltages(bus_name=node,phases=1,kw_inj=0,kvar_inj=-100)
                vph_perturbed_yorder = [vph_perturbed[i] for i in nodes] 
            else:
                raise Exception('Invalid dependent variable')

            if(indp_var == 'vmag'): #Vmag sensitivities
                S_matrix[:,col_idx] = (np.abs(np.asarray(vph_perturbed_yorder))-np.abs(np.asarray(vph_base_yorder)))/(100*100)
            elif(indp_var == 'theta'): #Angle sensitivities
                S_matrix[:,col_idx] = (np.angle(np.asarray(vph_perturbed_yorder))-np.angle(np.asarray(vph_base_yorder)))/(100*100)
            else:
                raise Exception("Invalid independent variable")
        
        return S_matrix,nodes,vph_base



    def __get_perturbed_bus_voltages(self,bus_name,phases,kw_inj,kvar_inj):
        """
        Gets a vector of perturbed BUS voltages
        """
        self.compile_dss(self.redirects)
        self.dss.run_command('Set Controlmode = STATIC')
        self.__set_perturbed_injection(bus_name,phases,kw_inj,kvar_inj)
        self.dss.run_command('solve')
        return self.dss.Circuit.AllBusMagPu()

    def __get_perturbed_nodal_voltages(self,bus_name,phases,kw_inj,kvar_inj):
        """
        Gets DICTIONARY of perturbed NODAL voltages 
        after a perturbed real or reactive power injection
        """
        self.compile_dss(self.redirects)
        self.dss.run_command('Set Controlmode = STATIC')
        self.__set_perturbed_injection(bus_name,phases,kw_inj,kvar_inj)
        self.dss.run_command('solve')
        return self.get_node_voltages()


    def __set_perturbed_injection(self,bus_name,phases,kw_inj,kvar_inj):
        """Places a perturbing injection on a bus of interest"""
        bus_name = str(bus_name)
        phases = str(phases)
        kw_inj = str(kw_inj)
        kvar_inj = str(kvar_inj)
        injection_name = bus_name + "_static_inj" # define the injection name
        err = self.dss.run_command(
            "New Load.{injection_name} Bus1={bus_name} Phases={phases} kW ={kw_inj} kvar={kvar_inj}".format(
            injection_name='load'+str(injection_name),
            bus_name=bus_name,
            phases = phases,
            kw_inj = kw_inj,
            kvar_inj = kvar_inj
            ))
        if(self.verbose):
            print(err)
        elif(err != ''):
            warnings.warn("Perturbed injection failed, OpenDSS returned:")
            print(err)

    #TODO: Implment this function
    def __truncate_sensitivities(self,S,nodes_all,vph_base):
        """
        Selects the rows and columns of chosen bus indeces of the internal nodal
        sensitivity matrices. 
        Returns as a dictionary
        """
    ##############################################
        pass
    ##############################################
        nodes = self.dss.Circuit.YNodeOrder()

        indeces = []
        for i,node in enumerate(nodes):
            node = self.split_node_name(node)
            if node not in self.pq_nodes:
                indeces.append(i)
        indeces = np.asarray(indeces)

        #Truncate the basecase phasor voltage dictionary to only include PQ buses.
        pq_vph_base = [vph_base[node] for node in self.pq_nodes]

        return S[indeces,:][:,indeces],self.pq_nodes,pq_vph_base


