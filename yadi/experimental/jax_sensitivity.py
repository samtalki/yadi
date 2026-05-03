# Experimental: incomplete research code, API may change.
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


#TODO
def train(v,p,q,lamb0=1e-5):
    """Given AMI dataset matrices v,p,q of size (M x N), please complete the following tasks:
        1.Form finite difference matrices DV,DP,DQ of size ((M-1)xN)
        2. Solve the regression problem with your favorite gradient descent method (notice that I've formed the gradient for you below).
        
    Next steps:
        1. Use this model to solve the hc problem: 
            - Given the sensitivities: 
            - What is the largest generation/demand dx so that we can guarantee a v+ dv<1.05? 
            - What about a v + dv >0.95? 
    """
    m,n = v.shape
    #Take finite differences
    dv,dp,dq = fdiff(v),fdiff(p),fdiff(q)
    dx = jnp.vstack((dp.T,dq.T,jnp.ones(dp.shape[1]))).T
    init_sens = jnp.transpose(jnp.dot(jnp.linalg.pinv(dx),dv))
    dvdp,dvdq = init_sens[:,:n],init_sens[:,n:]
    sol = ridge_reg_solution(
        init_sens=(dvdp,dvdq),
        lamb=lamb0,
        X=dx,y=dv
    )
    (Svp,Svq) = sol
    return Svp,Svq

#TODO
def test(v,p,q):
    """Given test AMI dataset matrices v,p,q of size (M x N), please complete the following tasks:
        1. Evalute the performance of the learned sensitivity model
        2. Compute various error metrices
    """
    pass

def predict(sens,v0,X):
    """Finite difference prediction of the voltage magnitudes"""
    #offset v0, vmag-active power sensitivities Svp, and vmag-reactive power sensitivities Svq
    Svp,Svq = sens 
    #Finite difference measurements
    p,q = X
    return v0 + jnp.dot(Svp,p) + jnp.dot(Svq,q)  #Multiple linear regression model.
def ridge_reg_objective(sens, lamb, X, y):
    residuals = jnp.dot(X, sens) - y
    return jnp.mean(residuals ** 2) + 0.5 * l2_regularizer(sens,lamb)

def ridge_reg_solution(init_sens,lamb, X, y,maxiter=10000,implicit_diff=True):
    """
    Returns the ridge regression sensitivity model solution for a given init_sens, lamb, X, y, and other arguments.
    """
    gd = GradientDescent(fun=ridge_reg_objective, maxiter=maxiter, implicit_diff=implicit_diff)
    sol= gd.run(init_sens, l2reg=lamb, X=X, y=y).params
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
    return jac(lamb,X,y)

def validation_loss(l2reg):
  sol = ridge_reg_solution(l2reg, X_train, y_train)
  residuals = jnp.dot(X_val, params) - y_val
  return jnp.mean(residuals ** 2)

def loss_l2reg_jacobian(l2reg):
    df = lambda l2reg: grad(validation_loss)(l2reg)


def l2_regularizer(sens,lamb):
    """The standard l2 regularization term, with optional separate lambdas for active/reactive power voltage magntiude sensitivities"""
    if(len(lamb)>1):
        assert len(lamb) == len(S)
        (l_vp,l_vq) = lamb
        (Svp,Svq) = sens
        regularization = l_vp*jnp.dot(Svp**2) + l_vq*jnp.dot(Svq**2)
    elif(len(lamb)==1 and len(sens)==2):
        (Svp,Svq) = sens
        regularization = lamb*jnp.dot(Svp**2) + lamb*jnp.dot(Svq**2)
    else:
        regularization = 0
    return regularization

