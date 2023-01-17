"""
Model-derived Voltage-Constraint Hosting Capacity Analysis (HCA) computation.
@author: Samuel Talkington
01/16/2023
"""
import numpy as np
import yadi.dss.model as model
import warnings

class DSS_VC_HCA(model.DSS_Data):

    def __init__(self,redirects,v_max=1.05,delta_kw_inj=1,kw_inj_max=500,max_hc_iter=50,verbose=True) -> None:
        """
        model-based Voltage constrained HCA analysis tool
        
        Params:
            - redirects: list of model files
            - v_max: maximum voltage magnitude in per unit
            - delta_kw_inj: increment of active power in kW
            - max_hc_iter: maximum amount of hc iterations for every node
            - verbose: verbose bool
        """
        super().__init__(redirects,verbose=verbose)
        
        #--- hc process params
        self.v_max = v_max
        self.delta_kw_inj = delta_kw_inj
        self.kw_inj_max = kw_inj_max
        self.max_hc_iter=max_hc_iter
        
        #--- nodes and power factor params
        self.nodes = self.dss.Circuit.YNodeOrder()
        self.n_nodes = len(self.nodes)
        self.pfs = np.ones(self.n_nodes)


    def get_iterative_hc(self,pfs=None):
        """
        Get the vector of maximum kw hosting capacities capacities for every bus with a load
        """
        if pfs is not None:
            self.pfs = pfs

        #Precompile the DSS file
        self.compile_dss(self.redirects)
        self.dss.run_command('solve')
        
        #--- All nodes
        nodes = self.dss.Circuit.YNodeOrder()
        loads = self.dss.Loads
        
        #--- Base voltages and idx where initial overvoltages do not occur
        vpu_base = np.copy(self.dss.Circuit.AllBusMagPu())
        sub_overvoltage_idx = [i for i in range(len(vpu_base)) if vpu_base[i]<self.v_max]

        #--- HC computations
        hc = np.zeros(self.n_nodes)
        for node_idx,node in enumerate(self.nodes):
            if vpu_base[node_idx] > 1.05:
                hc[node_idx] = 0
                continue
            else:
                for kw_inj in [-1*i for i in np.arange(start=1,stop=self.kw_inj_max,step=self.delta_kw_inj)]:
                    vph_perturbed = self.__get_perturbed_nodal_voltages(bus_name=node,phases=1,kw_inj=kw_inj,kvar_inj=0)
                    vpu_perturbed = np.asarray(self.dss.Circuit.AllBusMagPu())
                    if np.any(vpu_perturbed[sub_overvoltage_idx] > self.v_max):
                        hc[node_idx] = kw_inj
                        break
        return hc

    def get_max_kw(self,bus,pf=1):
        """
        Gets the maximum kW for a given bus given a power factor setting pf
        """
        def increment_kw(bus,pf=pf):
            """
            Internal kW incrementing function
            """
            pass
        pass

    @staticmethod
    def get_q_constraints(pf):
        """
        Computes q constraints based on vector of pf settings
        """
        pass


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
    