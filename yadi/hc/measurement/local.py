#!/usr/bin/env python
""" 
Methods for computing localized hosting capacity with linear voltage magnitude sensitivity models.
"""
import yadi.sens.data_linear as linear
import yadi.utils as utils
import numpy as np
from numba import jit

@jit(forceobj=True)
def pq_fit(p,q,delta,lambd,absolute_value,filter_var):
    """
    Explain deviations of reactive power via deviations of active power
    Coefficients are [q/p,q0]^T
    """
    X = np.vstack((p.T,np.ones(p.shape[0]).T)).T
    dqdp =linear.lsq_filtered(X,q,delta=delta,lambd=lambd,absolute_value=absolute_value,filter_var=filter_var)
    return dqdp

@jit(forceobj=True)
def estimate_pf(p,q,delta=None,lambd=None,absolute_value=None,filter_var=None):
    """
    Estimate power factor by fitting q/p and computing
    cos(arctan(dqdp))
    """
    s_coef = pq_fit(p,q,delta,lambd,absolute_value=absolute_value,filter_var=filter_var)
    return np.cos(np.arctan(s_coef[0]))


@jit(parallel=True,forceobj=True)
def analyze_hosting_capacity(V_test,V_train,Q,P,delta,lambd,filter_var,absolute_value,pf_inj=None):
    """
    Hosting capacity analysis scheme applying excitation filter and least squares regression
    """
    HC = [] #HC list with many methods compared
    v_S = [] #V-to-P and Q-to-P sensitivity params
    pq_S,pf = [],[] #Q-to-P sensitivity params
    X_filtered,v_filtered = [],[] #Filtered data output
    
    for study_idx,(p,q,v) in enumerate(zip(P.T,Q.T,V_train.T)):

        #Make training deviations for this node
        X = np.vstack((p.T,q.T,np.ones(p.shape[0]).T)).T #with offsetss
        #X = np.vstack((p.T,q.T)).T #Or, optionally, without offsets
        
        #Preprocess the data
        X,v = utils.drop_nan_rows(X,v) #Drop missing rows
        DX,dv = utils.diff(X,v) #Gradients of finite diffs

        #Filter and save data
        DX,dv = linear.excitation_filter(DX,dv,quantile=delta,absolute_value=absolute_value,filter_var=filter_var)
        v_filtered.append(dv)
        X_filtered.append(DX)
        
        #Solve sensitivity models
        v_S.append(linear.pinv(DX,lambd)@dv) #dvdp and dvdq
        pq_S.append(pq_fit(p=DX[:,0],q=DX[:,1],delta=delta,lambd=lambd,absolute_value=absolute_value,filter_var=filter_var)) #dqdp fit
        pf.append(estimate_pf(p=DX[:,0],q=DX[:,1],delta=delta,lambd=lambd,absolute_value=absolute_value,filter_var=filter_var)) #Compute pf
    
        #Get test instantaneous voltage
        k_max = np.argmax(V_test[:,study_idx])
        v_k_max,q_k_max = V_test[k_max,study_idx],q[k_max] #V and Q observed at the maximum voltage timepoint
        #v_plus_q_max = np.max(v + q*v_S[-1][1])
        
        #Prepare optional estimate power factor
        if(pf_inj is None):
            pf_hc = pf[-1]
        else:
            pf_hc = pf_inj

        #Compute HC
        HC.append(
            compute_hc(v_k_max=v_k_max,q_k=q_k_max,
            dvdp=v_S[-1][0],dvdq=v_S[-1][1],dqdp=pq_S[-1][0],pf=pf_hc
            )
        )
        
        #Force negative HC values to zero
        for i,hc_i in enumerate(HC[-1]):
            if(hc_i<=0):
                HC[-1][i] = 0

    return np.asarray(HC),v_S,pq_S,X_filtered,v_filtered,pf


@jit
def compute_hc(v_k_max,dvdp,dvdq,dqdp,pf,q_k,v_lim=1.05):
    """
    Various definitions of hosting capacity to be compared
    """
    qp_HC = (v_lim-v_k_max)/(dvdp+dvdq*dqdp) #HC via active/reactive power sensitivities in tandem
    signed_hc = (v_lim-v_k_max)/(dvdp-np.sign(q_k)*dvdq*dqdp) #HV via P/Q and switchin signs of complex plane
    pf_scaled_hc = ((v_lim-v_k_max)/(dvdp-np.sign(q_k)*dvdq*dqdp))/pf #PF scaling using learned power factor
    arbitrary_pf_scaled_hc = ((v_lim-v_k_max)/(dvdp+dvdq*dqdp))/0.7 #arbitrary 0.7 fixed pf injection
    p_HC = (v_lim-v_k_max)/(dvdp) #HC via active power sensitivities only    
    return [qp_HC,signed_hc,pf_scaled_hc,arbitrary_pf_scaled_hc,p_HC]
