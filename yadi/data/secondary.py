import numpy as np
import opendssdirect as dss
import yadi.dss.sensitivity as sensitivity


class DSS_Secondaries(sensitivity.DSS_Sensitivities):
    """
    WIP: "distributed" power flow solves exploiting knowledge of secondary topology
    """

    def __init__(self,redirects,verbose=False) -> None:
        super().__init__(redirects,verbose)
        #Full size sensitivity matrices with all secondaries
        self.svp = self.get_spv()
        self.svq = self.get_sqv()
        self.secondaries = None #Dictionary of secondary matrices
        
    def get_secondary_data(self,n_secondaries = 10):
        """
        Make slices of the node indexes of the currently loaded opendss feeder mapping to the secondary networks
        Params:
            n_secondaries: Number of secondary groups in the feeder
        """
        #node_names = dss.Circuit.AllNodeNames()[33:]
        self.secondary_data = {}
        node_names = dss.Circuit.AllNodeNames()
        sec_idxs = [str(i) for i in range(1,n_secondaries+1)] #Group index of the secondary names
        for sec_idx in sec_idxs:
            node_idxs_in_sec,node_names_in_sec = [],[] 
            for node_idx,node_name in enumerate(node_names):
                if('sec' + sec_idx + "_" in node_name): #Hacky way to separate by node name
                    node_idxs_in_sec.append(node_idx)
                    node_names_in_sec.append(node_name)
            self.secondary_data['sec' + sec_idx] = {
                "node_idxs":node_idxs_in_sec,
                "node_names":node_names_in_sec
            }
        return self.secondary_data         

    def make_sensitivities_secondaries(self,n_secondaries=10):
        """
        Construct the sensitivity matrices for all of the secondary groupings created by get_secondary_data
        """
        secondary_data = self.get_secondary_data()
        svp0,svq0 = self.svp['matrix'],self.svq['matrix']
        for S in [svp0,svq0]:
            for i,(sec_name,d) in enumerate(secondary_data.items()): 
                #Make the secondary sensitivity matrix
                node_idxs = d["node_idxs"]
                sec_S = S[np.ix_(node_idxs,node_idxs)]
                secondary_data[sec_name]["S"] = sec_S #save
        return None

    def make_secondaries(self,Ynet,n_secondaries = 10):
        """
        Make slices of the node indexes of the currently loaded opendss feeder mapping to the secondary networks
        Params:
            Ynet: nodal admittance matrix
            n_secondaries: Number of secondary groups in the feeder
        """
        #node_names = dss.Circuit.AllNodeNames()[33:]
        secondaries = {}
        node_names = dss.Circuit.AllNodeNames()
        sec_idxs = [str(i) for i in range(1,n_secondaries+1)] #Group index of the secondary names
        for sec_idx in sec_idxs:
            node_idxs_in_sec,node_names_in_sec = [],[] 
            for node_idx,node_name in enumerate(node_names):
                if('sec' + sec_idx + "_" in node_name): #Hacky way to separate by node name
                    node_idxs_in_sec.append(node_idx)
                    node_names_in_sec.append(node_name)
            #Make secondary ybus
            Ysec = Ynet[np.ix_(node_idxs_in_sec,node_idxs_in_sec)]
            
            #Format the names of the nodes
            formated_names = []
            for name in node_names_in_sec:
                name = name[3:] #Remove the "bus" at the beginning of the name
                name = name[-3:]
                #name = name.replace(
                #    "sec{sec_idx}_".format(sec_idx=sec_idx),
                #    "")
                #name =  "b" + str(int(np.mod(sec_idx,3)))
                #name = name.replace(".","ph") #Replace the dot syntax
                name = name.replace(".1",".A")
                name = name.replace(".2",".B")
                name = name.replace(".3",".C")
                formated_names.append(name)
            secondaries['sec' + sec_idx] = {
                "node_idxs":node_idxs_in_sec,
                "node_names":formated_names,
                "Ysec":Ysec
            }
        return secondaries         

def calc_vph_active_sensitivities(Y,vph):
    #Number of buses and equations
    n_bus = len(vph)
    n_equ = 2*n_bus
    assert Y.shape == (n_bus,n_bus)
    
    #Sensitivity coefficients
    dvp,dvp_c = np.zeros_like(Y),np.zeros_like(Y)
    dvp_real,dvp_imag = np.real(dvp),np.imag(dvp)
    dvp_c_real,dvp_c_imag = np.real(dvp_c),np.imag(dvp_c)
    
    #Quantities for building equations
    conj_coef = Y @ vph

    #LHS of equation
    lhs = np.vstack((np.eye(3),np.eye(3)))

    #RHS of equation
    A = np.block([
        [np.diag(Y@vph), np.diag(np.conj(vph))@Y],
        [-np.diag(np.conj(vph))@Y, np.diag(np.conj(vph))@Y]
        ])
    X = np.linalg.inv(A)@lhs
    return X

    # for l in range(n_bus):
    #     for i in range(n_bus): #N_injection systems of N_bus equations 
    #         lhs = 1 if i == l else 0 #lhs of the equation
            
    #         #Coefficients for the standard sensitivity and the conjugate
    #         dvp_coef = np.dot(Y[i,:],vph)
    #         dvp_conj_coef = np.dot()

    #         #Store the resulting coefficients
    #         dvp[i,l] = 
    #         dvp_c[i,l] = 


def calc_vph_reactive_sensitivities(Y,vph):
    pass
    