"""
OpenDSS EV Simulation Data Structure
@author: Samuel Talkington and Alex Reyna
May 24th, 2022
"""
import yadi.dss.model as model 
import yadi.sens.model_perturb as perturb
import jax.numpy as jnp

#Optional: Turn off complex value warnings
#warnings.simplefilter("ignore", np.ComplexWarning)

class OpenDSS_EV(perturb.OpenDSS_Sensitivities):

    def __init__(self, redirects, verbose=False,unity_pf=True):
        super().__init__(redirects, verbose)
        self.unity_pf = unity_pf
        self.groups = None #Group slices for the vehicles

    def predict(self,nodes):
        """
        Predict the EV hosting capacity (maximum amount of demand) for each node 
        """
        n_nodes = len(nodes)
        eta = jnp.zeros_like(n_nodes)
        if(self.unity_pf):
            svp_res = self.get_svp()
            svp = svp_res["matrix"]
            nodes = svp_res["nodes"]
            vph_base = svp_res["vph_base"]
            
        
        return eta


    def set_groups(self,group_slices):
        """Set the groups for"""
        for slice in group_slices:
            self.