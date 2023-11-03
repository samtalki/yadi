## Conservative linear approximations
import yadi.dss.model as model
import opendssdirect as dss
import numpy as np
import cvxpy as cp
import warnings 

class CLA:
    """
    Conservative linear approximation : regression within bounds 
    """

    def __init__(self,verbose=True,maxiters=1000,solver="SCS") -> None:
        self.verbose = verbose
        self.maxiters=maxiters
        self.solver=solver
        self.params = (None,None) # parameters of the linear approximation 
        self.intercept = None # intercept of the linear approximation
        self.dvp = None # sensitivity of vmag wrt active power
        self.dvq = None # sensitivity of vmag wrt reactive power
        self.dv2p = None # sensitivity of vmag^2 wrt active power
        self.dv2q = None # sensitivity of vmag^2 wrt reactive power

    def fit(self,X,Y,ub=False,lb=False):
        """
        Make the conservative linear approximation model.
        The program is a constrained l1-norm approximation problem.
        Parameters:
            - X (np.array): matrix of independent variables
            - Y (np.array): matrix of dependent variables
            - ub (np.array): upper bounds for the approximation (if CLA is an underestimate)
            - lb (np.array): lower bounds for the approximation (if CLA is an overestimate)
        """
        (m,n) = X.shape
        assert m==Y.shape[0], "X and Y must have the same number of rows"

        # Define the variables
        a = cp.Variable(n) # coefficients
        b = cp.Variable(1) # intercept
        # Define the constraints
        
        constraints = []
        if ub is not None or lb is not None:               
            if ub is not None:
                constraints.append(X@a + b*np.ones(m) <= ub)
            if lb is not None:
                constraints.append(X@a + b*np.ones(m) >= lb)
        else:
            warnings.warn("No upper or lower bounds specified. Using unconstrained approximation.")

        # Define and solve the problem        
        obj = cp.norm(Y - (X@a + b*np.ones(m)),1)
        prob = cp.Problem(cp.Minimize(obj),constraints)
        prob.solve(
            solver=self.solver,
            verbose=self.verbose,
            max_iters=self.maxiters)

        return a.value,b.value # return the coefficients and intercept
        

    def fit_pq(self,P,Q,V,ub=None,lb=None):
        """
        Make a conservative linear approximation of the relationship between a quantity and P,Q
        Parameters:
            - V (np.array): (M x 1) matrix of vmag^2 values
            - P (np.array): (M x n) matrix of active power values
            - Q (np.array): (M x n) matrix of reactive power values
        """
        (m,n) = P.shape
        (mQ,nQ) = Q.shape
        assert m==mQ, "P and Q must have the same number of rows"
        assert m==V.shape[0], "V must have the same number of rows as P and Q"

        a,b = self.fit(
            np.hstack((P,Q)),
            V,
            ub=ub,
            lb=lb)
        self.dv2p = a[:n]
        self.dv2q = a[n:]
        self.intercept = b
        return self.dv2p,self.dv2q,self.intercept
