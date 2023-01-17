##
# EV Hosting capacity analysis (Data driven/sensitivity based and model-based)
##

import numpy as np
import jax.numpy as jnp
from jax import jit,grad,vmap

#Abstract, high-level function
def EVHC():
    pass

#EV hosting capacity using the network model
def ModelEVHC(circuit):
    pass

#EV hosting capacity using the sensitivity matrices and the AMI measurements
def SensitivityEVHC(S):
    Dvp,Dvq = S
    
    DVP = jnp.zeros() #Voltage magnitude-active power sensitivity matrix
    DVQ = jnp.array() #Voltage magnitude-reactive power sensitivity matrix


    def predict_hc(x,v_lim):
        pass


