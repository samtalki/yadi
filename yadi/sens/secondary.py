import numpy as np
import opendssdirect as dss
from numba import jit

@jit
def calc_vph_active_sensitivities(Y,vph):
    #Number of buses and equations
    n_bus = len(vph)
    n_equ = 2*n_bus
    assert Y.shape == (n_bus,n_bus)
    
    #Sensitivity coefficients
    dvp,dvp_c = np.zeros_like(Y),np.zeros_like(Y)
    dvp_real,dvp_imag = np.real(dvp),np.imag(dvp)
    dvp_c_real,dvp_c_imag = np.real(dvp_c),np.imag(dvp_c)
    
    #Quantities for building equations
    conj_coef = Y @ vph

    #LHS of equation
    lhs = np.vstack((np.eye(3),np.eye(3)))

    #RHS of equation
    A = np.block([
        [np.diag(Y@vph), np.diag(np.conj(vph))@Y],
        [-np.diag(np.conj(vph))@Y, np.diag(np.conj(vph))@Y]
        ])
    X = np.linalg.inv(A)@lhs
    return X

    # for l in range(n_bus):
    #     for i in range(n_bus): #N_injection systems of N_bus equations 
    #         lhs = 1 if i == l else 0 #lhs of the equation
            
    #         #Coefficients for the standard sensitivity and the conjugate
    #         dvp_coef = np.dot(Y[i,:],vph)
    #         dvp_conj_coef = np.dot()

    #         #Store the resulting coefficients
    #         dvp[i,l] = 
    #         dvp_c[i,l] = 


def calc_vph_reactive_sensitivities(Y,vph):
    pass
    