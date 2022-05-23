#---
# General purpose linear sensitivity model in Jax for any steady-state time series electrical network.
# 
#   Summary:
#   Relate K MxN matrices of M time series measurements from N nodes 
#   Through K sensitivity matrices S_1,...,S_k and optionally offset x_0 such that
#   Y = X_0 + S_1@X_1 + S_2@X_2 +...+ S_kX_k

#   Current Iteration:
#   Relate (MxN) DV with (MxN) DP and (MxN) DQ <--> (Mx2N) DX
#   DV = V0 + Svp@DP + Svq@DQ 
#---

#---
#Helpful reading:
#- How to Think in JAX: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html
#- The JAX Autodiff Cookbook: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
#---
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)


def predict(S,v0,dx):
    """Finite difference prediction of the voltage magnitudes"""
    #Finite difference measurements
    dp,dq = dx
    #offset v0, vmag-active power sensitivities Svp, and vmag-reactive power sensitivities Svq
    Svp,Svq = S 
    return v0 + jnp.dot(Svp,dp) + jnp.dot(Svq,dq)  #Multiple linear regression model.


def fdiff(A,norm=1):
    """Given an (MxN) data matrix A, compute the finite diference matrix DA
    Params:
        norm: the amount to normalize A(k+1)-A(k) by
    """
    (M,N) = A.shape
    DA = jnp.divide(
        jnp.diff(A,axis=0),
        norm)
    assert DA.shape == (M-1,N)
    return DA
    

def train(v,p,q):
    """Given AMI dataset matrices v,p,q of size (M x N), please complete the following tasks:
        1.Form finite difference matrices DV,DP,DQ of size ((M-1)xN)
        2. Solve the regression problem with your favorite gradient descent method (notice that I've formed the gradient for you below).
        
    Next steps:
        1. Use this model to solve the hc problem: 
            - Given the sensitivities: 
            - What is the largest generation/demand dx so that we can guarantee a v+ dv<1.05? 
            - What about a v + dv >0.95? 
    """
    
    # Training loss is the negative log-likelihood of the training examples.
    def l2_loss(dv,S,v0, dx):
        dv_hat = predict(S, v0, dx)
        resid_sq = jnp.linalg.norm(dv_hat - dv)**2
        return jnp.sum(resid_sq)
    
    #TODO: Add an L2 regularization term
    def l2_reg(S):
        pass

    #Take finite differences
    dv,dp,dq = fdiff(v),fdiff(p),fdiff(q)
    grad = jit(grad(l2_loss))
    return S,v0
    


# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = jnp.array([True, True, False, True])

# Initialize random model coefficients
key, W_key, b_key = random.split(key, 3)
W = random.normal(W_key, (3,))
b = random.normal(b_key, ())