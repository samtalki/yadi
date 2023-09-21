#Data-driven hosting capacity analysis for groups
# Compute an N-dimensional vector of nodal hosting capacities eta
import jax.numpy as jnp
from jax import jit,grad,vmap
from jaxopt.projection import projection_non_negative

def predict_max_unity_gen(sens,v,v_lim=1.05):
    """
    Compute a vector of nodal hosting capabilities "eta" at time t for node j.
    Params:
        sens: (Svp,Svq), each Sensitivity matrix of size (n_nodes x n_nodes). If active/reactive included this is (n_nodes,2*n_nodes)
        v (ndarray): size (n_nodes): Vector of INSTANTANEOUS voltage measurements.
        v_lim (int),(ndarray): size (n_nodes): Vector or integer of voltage limits
    """
    if(isinstance(v_lim,int)):
        v_lim = v_lim*jnp.ones(N)
    M,N = v.shape
    (Svp,Svq) = sens
    v_dist = v_lim - v
    p_pls = jnp.dot(jnp.linalg.inv(Svp),v_dist)
    return projection_non_negative(p_pls)
    
def objective_max_unity_gen(sens,X,v,v_lim=1.05):
    """
    Compute the objective function value for the hc problem
    Params:
        - given sensitivity model `sens`, 
        - power fluctuations `X`, 
        - instantaneous voltage `v`,
        - voltage limit(s) `v_lim`
    """
    if(isinstance(v_lim,int)):
        v_lim = v_lim*jnp.ones(N)
    M,N = v.shape
    (Svp,Svq) = sens
    (dp,dq) = X
    v_hat = jnp.dot(Svp,dp) + jnp.dot(Svq,dq)
    return jnp.linalg.norm((v+v_hat)-v_lim)
    

def unity_capability(sens,v,v_ub=1.05):
    M,N = v.shape
    (Svp,Svq) = sens
    if(isinstance(v_ub,int)):
        v_ub = v_ub*jnp.ones(N)
    capab = jnp.linalg.inv(S)*(v_ub-v)
    return jnp.clip(a=capab,a_min=0,a_max=5e3) #Clip the HC values to make sense

#Jit compiled/vectorized mapping of the group capabilitiy computation
unity_capability_vmap = jit(vmap(unity_capability,in_axes=(None,0,None),out_axes=0))


