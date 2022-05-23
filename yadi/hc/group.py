#Data-driven hosting capacity analysis for groups
# Compute an N-dimensional vector of nodal hosting capacities eta

import jax.numpy as jnp
from jax import jit,grad,vmap

@jit
def unity_capability(S,v,v_ub=1.05):
    """
    Compute a vector of nodal hosting capabilities "eta" at time t for node j.
    Params:
        S: (ndarray), size (n_injections,n_nodes) Sensitivity matrix. If active/reactive included this is (2*n_nodes,n_nodes)
        v (ndarray): size (n_nodes): Vector of INSTANTANEOUS voltage measurements.
        v_lim (int),(ndarray): size (n_nodes): Vector or integer of voltage limits
    """
    M,N = v.shape
    if(isinstance(v_ub,int)):
        v_ub = v_ub*jnp.ones(N)
    capab = jnp.linalg.inv(S)*(v_ub-v)
    return jnp.clip(a=capab,a_min=0,a_max=5e3) #Clip the HC values to make sense

#Jit compiled/vectorized mapping of the group capabilitiy computation
unity_capability_vmap = jit(vmap(unity_capability,in_axes=(None,0,None),out_axes=0))


