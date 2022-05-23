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
#JAX imports
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import jacobian
from jax import random
#RNG key
key = random.PRNGKey(0)
#Gradient descent solvers
from jaxopt import GradientDescent
from jaxopt import ProjectedGradient
#Finite difference function
from yadi.utils import fdiff

def predict(S,v0,dx):
    """Finite difference prediction of the voltage magnitudes"""
    #Finite difference measurements
    dp,dq = dx
    #offset v0, vmag-active power sensitivities Svp, and vmag-reactive power sensitivities Svq
    Svp,Svq = S 
    return v0 + jnp.dot(Svp,dp) + jnp.dot(Svq,dq)  #Multiple linear regression model.
  
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
    def l2_loss(S,v0,dx,dv):
        """The training loss is the standard least-squares l2 loss."""
        dv_hat = predict(S, v0, dx)
        return jnp.linalg.norm(dv_hat - dv)**2

    def l2_reg(S,lamb):
        """The standard l2 regularization term, with optional separate lambdas for active/reactive power voltage magntiude sensitivities"""
        if(len(lamb)>1):
            assert len(lamb) == len(S)
            (l_vp,l_vq) = lamb
            (Svp,Svq) = S
            reg = l_vp*jnp.linalg.norm(Svp) + l_vq*jnp.linalg.norm(Svq)
        elif(len(lamb)==1 and len(S)==2):
            (Svp,Svq) = S
            reg = lamb*jnp.linalg.norm(Svp) + lamb*jnp.linalg.norm(Svq)
        else:
            reg = 0
        return reg
    
    def ridge_reg_objective(params, lamb, X, y):
        residuals = jnp.dot(X, params) - y
        return jnp.mean(residuals ** 2) + 0.5 * lamb * jnp.dot(params ** 2)

    def ridge_reg_solution(lamb, X, y,
        init_params,
        maxiter=10000,implicit_diff=True
    ):
        gd = GradientDescent(fun=ridge_reg_objective, maxiter=maxiter, implicit_diff=implicit_diff)
        sol= gd.run(init_params, l2reg=lamb, X=X, y=y).params
        return sol

    def solution_jacobian(arg,lamb,X,y,sol=None):
        """
        Compute the jacobian of the solution with respect to argnum evaluated at (params,state)
        """
        if sol is None:
            sol = ridge_reg_solution
        if(arg==0 or 'lamb' in arg):
            jac = lambda lamb,X,y : jacobian(sol, argnums=0)(lamb, X, y)
        elif(arg==1 or 'X' in arg): 
            jac = lambda lamb,X,y : jacobian(sol, argnums=1)(lamb, X, y)
        return jac



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