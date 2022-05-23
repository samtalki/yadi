import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

method_names = ["qp","signed_qp","est_pf_scaled_qp",'arbitrary_pf_scaled_qp',"p"]
method_descs = [
    r"$\hat{HC} = \frac{1.05-max_t v_t}{\frac{dv}{dp} + \frac{dv}{dq} \frac{dq}{dp}}$",
    r"$\hat{HC} = \frac{1.05 - max_t v_t}{\frac{dv}{dp} - sign(q_t) \frac{dv}{dq} \frac{dq}{dp}}$",
    r"$\hat{HC} = \frac{1.05 - max_t v_t}{\frac{dv}{dp} + \frac{dv}{dq} \frac{dq}{dp}} / \hat{pf}$",
    r"$\hat{HC} = \frac{1.05 - max_t v_t}{\frac{dv}{dp} + \frac{dv}{dq} \frac{dq}{dp}} / (pf=0.7)$",
    r"$\hat{HC} = \frac{1.05 - max_t v_t}{\frac{dv}{dp} }$"
]

def print_results(HC_1,HC_2,baseline):
    for method_name,hc_1i,hc2_i in zip(method_names,HC_1.T,HC_2.T):
        sc1_err,sc2_err = np.abs(hc_1i-baseline['all']),np.abs(hc2_i-baseline['day'])
        sc1_mae,sc2_mae = np.mean(sc1_err),np.mean(sc2_err)
        sc1_worst_idx,sc2_worst_idx = np.argmax(sc1_err),np.argmax(sc2_err)
        print("========================================Method {method} Summary: ".format(method=method_name))
        print('(sc1,sc2) mae: ({sc1_mae:.2f},{sc2_mae:.2f})'.format(sc1_mae=sc1_mae,sc2_mae=sc2_mae))
        print('(sc1,sc2) worst case: ({sc1worst:.2f},{sc2worst:.2f}) '.format(sc1worst=np.max(sc1_err),sc2worst=np.max(sc2_err)),' on indeces: ({sc1_worst_idx},{sc2_worst_idx})'.format(
            sc1_worst_idx=sc1_worst_idx,sc2_worst_idx=sc2_worst_idx))

def plot_results(HC_1,HC_2,baseline):
    cmap = plt.get_cmap('tab10')
    for method_name,desc,hc_1i,hc2_i in zip(method_names,method_descs,HC_1.T,HC_2.T): 
        sc1_err,sc2_err = np.abs(hc_1i-baseline['all']),np.abs(hc2_i-baseline['day'])
        sc1_mae,sc2_mae = np.mean(sc1_err),np.mean(sc2_err)
        fig,axes = plt.subplots(constrained_layout=True,figsize=(3.5*2,3.5*2/1.61828),ncols=2,sharey=True)
        axes[0].plot(baseline['all'],np.asarray(hc_1i),'s',alpha=0.25,c=cmap(2),markersize=3,label='$[HC,\hat{HC}]$')
        axes[0].plot(np.linspace(0,16),np.linspace(0,16),ls='--',c='k',label='Baseline 1-1')
        axes[1].plot(baseline['day'],np.asarray(hc2_i),'d',alpha=0.25,c=cmap(3),markersize=3,label='$[HC,\hat{HC}]$')
        axes[1].plot(np.linspace(0,16),np.linspace(0,16),ls='--',c='k',label='Baseline 1-1')
        plt.suptitle("method: {method}, {desc}".format(method=method_name,desc=desc))
        axes[0].set_title("sc 1: MAE:{mae:.3f} kW, worst: {worst:.3f} kW".format(
            mae=sc1_mae,worst=np.max(sc1_err)),fontsize=11)
        axes[1].set_title("sc 2: MAE:{mae:.3f} kW, worst: {worst:.3f} kW".format(
            mae=sc2_mae,worst=np.max(sc2_err)),fontsize=11)
        axes[0].set_xlabel('Model-Based HC (kW)')
        axes[1].set_xlabel('Model-Based HC (kW)')
        axes[0].set_ylabel('Model-Free HC (kW)')
        axes[0].set_xlim(-1,25)
        axes[0].set_ylim(-1,25)
        axes[1].set_xlim(-1,25)
        axes[1].set_ylim(-1,25)
        axes[0].legend()
        axes[1].legend() 

def load_data(data_path):
    AMI = ami.AMIData(data_path=data_path)
    data,day_data = AMI.get_datasets(),AMI.get_daytime_datasets()
    diff,day_diff = AMI.static_differentiate(data['V'],data['P'],data['Q']),AMI.static_differentiate(day_data['V'],day_data['P'],day_data['Q'])
    baseline = AMI.hc_baseline
    day_idxs = AMI.daytime_mask
    return {'data':data,'day_data':day_data,'diff':diff,'day_diff':day_diff,'baseline':baseline,'day_idx':day_idxs}

def drop_nans(X):
    nan_X = np.isnan(X)
    not_nan_X = ~nan_X
    X_new = X[not_nan_X]
    return X_new

def drop_nan_rows(X,v):
    Xv = np.vstack((X.T,v.T)).T
    rows_with_nans = np.isnan(Xv).any(axis=1)
    Xv_filtered = Xv[~rows_with_nans]
    X_filtered,v_filtered = Xv_filtered[:,:-1],Xv_filtered[:,-1]
    return X_filtered,v_filtered

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