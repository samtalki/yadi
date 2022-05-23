#--- 
# Sensitivity models relating to electric vehicle analysis (demand, undervoltage problems)
#---

import jax.numpy as jnp
from jax import jit,grad,vmap


def predict_max_unity_gen(S,v_0,v_lim):
    (Svp,Svq) = S
    v_dist = v_lim - v_0
    p_pls = jnp.dot(jnp.linalg.inv(Svp),v_dist)
    p_pls[p_pls<0] = 0
    return p_pls

def predict_max_unity_dem(S,v0,v_lim):
    """
    Predict the maximum demand for a given sensitivity model S
    Params:
        - S: (Svp,Svq) tuple of (N x N) voltage magnitude sensitivity matrices to active and reactive power
        - v0: (N x 1) vector of voltages observed now ("offset") 
    """
    (Svp,Svq) = S
    
    pass

