import numpy as np
import jax.numpy as jnp
from jax import jit,grad,vmap

def EVHC():
    pass



def SensitivityEVHC(circuit):
    
    DVP = jnp.zeros() #Voltage magnitude-active power sensitivity matrix
    DVQ = jnp.array() #Voltage magnitude-reactive power sensitivity matrix


    def predict_hc(x,v_lim):
        

    def
