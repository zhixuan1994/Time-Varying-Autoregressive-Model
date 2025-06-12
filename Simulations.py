import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
import warnings
from scipy.optimize import minimize, NonlinearConstraint
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import Time_Varying_Vector_Autoregressive
warnings.filterwarnings("ignore", category=UserWarning)

# Visualization and show the statistics results
def MAE_vis_data(TV_VAR_pred, y_true, r=3, H=3):
    mae_res = []
    for j in range(H):
        pred_given_r_mae = []
        for r_ind in range(r):
            pred_j_H, y_t_temp = [], []
            for i in range(len(TV_VAR_pred)):
                pred_j_H.append(TV_VAR_pred[i][r_ind][j])
                y_t_temp.append(y_true[i][j])
            pred_given_r_mae.append(mean_absolute_error(pred_j_H, y_t_temp))
        mae_res.append(pred_given_r_mae)
    return mae_res

def MAE_statis(y_pred, y_true, r=3, H=3):
    mean_res, var_res, min_res, max_res = [], [], [], []
    
    for j in range(H):
        mean_r_mae, var_r_mae, min_r_mae, max_r_mae = [], [], [], []
        for r_ind in range(r):
            pred_j_H, y_t_temp = [], []
            for i in range(len(y_pred)):
                pred_j_H.append(y_pred[i][r_ind][j])
                y_t_temp.append(y_true[i][j])
            pred_j_H = np.array(pred_j_H)
            y_t_temp = np.array(y_t_temp)
            mean_r_mae.append(np.mean(np.abs(pred_j_H - y_t_temp)))
            var_r_mae.append(np.var(np.abs(pred_j_H - y_t_temp), ddof=1))
            min_r_mae.append(np.min(np.abs(pred_j_H - y_t_temp)))
            max_r_mae.append(np.max(np.abs(pred_j_H - y_t_temp)))
        mean_res.append(mean_r_mae)
        var_res.append(var_r_mae)
        min_res.append(min_r_mae)
        max_res.append(max_r_mae)
    return mean_res, var_res, min_res, max_res

# Define the simulation time-varying matrix
def g_fun_p2(A, t):
    t = 1-t
    out = np.zeros(shape=A.shape)
    out[0,0] = np.min([A[0,0] + t, 1])
    out[0,1] = 1 - out[0,0]
    out[1,0] = np.min([A[1,0] + t, 1])
    out[1,1] = 1 - out[1,0]
    return out/2

def g_fun_p2_2(A, t):
    t = 1-t
    out = np.zeros(shape=A.shape)
    out[0,0] = np.max([A[0,0] - t, 0])
    out[0,1] = 1 - out[0,0]
    out[1,0] = np.min([A[1,0] + t, 1])
    out[1,1] = 1 - out[1,0]
    return out/2

def g_fun_p3(A, t):
    t = 1-t
    out = np.zeros(shape=A.shape)
    out[0,0] = np.min([A[0,0] + t,1])
    out[0,1] = np.min([A[0,1] + t,1])
    out[0,2] = 1 - out[0,0] - out[0,1]
    out[1,0] = np.min([A[0,0] + t,1])
    out[1,1] = np.max([A[1,1] - t,0])
    out[1,2] = 1 - out[1,0] - out[1,1]
    out[2,0] = np.max([A[0,0] - t,0])
    out[2,1] = np.min([A[0,1] + t,1])
    out[2,2] = 1 - out[2,0] - out[2,1]
    return out/2

def g_fun_p3_2(A, t):
    out = np.zeros(shape=A.shape)
    out[0,0] = np.max([A[0,0] - t,0])
    out[0,1] = np.max([A[0,1] - t,0])
    out[0,2] = 1 - out[0,0] - out[0,1]
    out[1,0] = np.max([A[0,0] - t,0])
    out[1,1] = np.min([A[1,1] + t,1])
    out[1,2] = 1 - out[1,0] - out[1,1]
    out[2,0] = np.min([A[0,0] + t,1])
    out[2,1] = np.max([A[0,1] - t,0])
    out[2,2] = 1 - out[2,0] - out[2,1]
    return out/2

def y_t_generate(y_0, t, p, A1_fixed, A2_fixed, A_type = 'p2'):
    eps_t = np.random.normal(0, 1, size=p)
    y_t = np.concatenate([y_0.reshape(1,-1), (y_0 + 0.1*np.random.normal(0, 1, size=p)).reshape(1,-1) ], axis=0)
    for i,j in enumerate(t[2:]):
        eps_t = 0.1*np.random.normal(0, 1, size=p)
        if A_type == 'p2':
            A1 = g_fun_p2(A1_fixed,j)
            A2 = g_fun_p2_2(A2_fixed,t[1+i])
        elif A_type == 'p3_1':
            A1 = g_fun_p3(A1_fixed,j)
            A2 = g_fun_p3_2(A2_fixed,t[1+i])
        y_t_next = y_t[-1].dot(A1) + y_t[-2].dot(A2) + eps_t
        y_t = np.concatenate([y_t, y_t_next.reshape(1,-1)], axis=0)
    return y_t

def y_t_generate2(y_0, t, p, A1_fixed, A2_fixed, A_type = 'p2'):
    eps_t = np.random.normal(0, 1, size=p)
    y_t = np.concatenate([y_0.reshape(1,-1), (y_0 + 0.1*np.random.normal(0, 1, size=p)).reshape(1,-1) ], axis=0)
    for i,j in enumerate(t[2:]):
        eps_t = 0.1*np.random.normal(0, 1, size=p)
        if A_type == 'p2':
            A1 = g_fun_p2(A1_fixed,j)
            A2 = g_fun_p2_2(A2_fixed,t[1+i])
        elif A_type == 'p3_1':
            A1 = g_fun_p3(A1_fixed,j)
            A2 = g_fun_p3_2(A2_fixed,t[1+i])
        y_t_next = y_t[-1].dot(A1) + y_t[-2].dot(A2) +t[1+i]/20+ eps_t
        y_t = np.concatenate([y_t, y_t_next.reshape(1,-1)], axis=0)
    return y_t

# Scenario 1
x0 = np.array([0,0])
t = np.arange(0, 1, 0.005)
ind_1 = -2
r_max = 3
p = 2
GAM_pred_p2, VAR_pred_p2, OKS_pred_p2, y_true_p2 = [], [], [], []
for i in tqdm(range(100)):
    A1_fixed = np.random.uniform(0,1,size=(2,2))
    A1_fixed = A1_fixed/np.sum(A1_fixed,axis=1).reshape(-1,1)
    A2_fixed = np.random.uniform(0,1,size=(2,2))
    A2_fixed = A2_fixed/np.sum(A2_fixed,axis=1).reshape(-1,1)
    GAM_pred_temp, OKS_pred_temp, VAR_pred_temp = [], [], []
    y_t = y_t_generate(x0, t, p, A1_fixed, A2_fixed, A_type='p2')
    for r in range(1, r_max):
        GAM_pred = Time_Varying_Vector_Autoregressive.TV_VAR_GAM(r, y_t[:ind_1], t[:ind_1])
        GAM_pred_temp.append(GAM_pred.GAM_main(2))
        OKS_pred = Time_Varying_Vector_Autoregressive.TV_VAR_OKS(r, y_t[:ind_1], t[:ind_1])
        OKS_pred_temp.append(OKS_pred.OKS_main(2))
        VAR_pred = Time_Varying_Vector_Autoregressive.TV_VAR_trad(r, y_t[:ind_1], t[:ind_1])
        VAR_pred_temp.append(VAR_pred.VAR_main(2))
    GAM_pred_p2.append(GAM_pred_temp)
    OKS_pred_p2.append(OKS_pred_temp)
    VAR_pred_p2.append(VAR_pred_temp)
    y_true_p2.append(y_t[ind_1:])
print(MAE_statis(GAM_pred_p2,y_true_p2,r=2, H=2))
print(MAE_statis(OKS_pred_p2,y_true_p2,r=2, H=2))
print(MAE_statis(VAR_pred_p2,y_true_p2,r=2, H=2))

# Scenario 2
x0 = np.array([0,0,0])
t = np.arange(0, 1, 0.005)
ind_1 = -2
r_max = 3
p=3
GAM_pred_p2_3, VAR_pred_p2_3, OKS_pred_p2_3, y_true_p2_3 = [], [], [], []
for i in tqdm(range(100)):
    A1_fixed = np.random.uniform(0,1,size=(3,3))
    A1_fixed = A1_fixed/np.sum(A1_fixed,axis=1).reshape(-1,1)
    A2_fixed = np.random.uniform(0,1,size=(3,3))
    A2_fixed = A2_fixed/np.sum(A2_fixed,axis=1).reshape(-1,1)
    y_t = y_t_generate(x0, t, p, A1_fixed, A2_fixed, A_type='p3_1')
    GAM_pred_temp, OKS_pred_temp, VAR_pred_temp = [], [], []
    for r in range(1, r_max):
        GAM_pred = Time_Varying_Vector_Autoregressive.TV_VAR_GAM(r, y_t[:ind_1], t[:ind_1])
        GAM_pred_temp.append(GAM_pred.GAM_main(2))
        OKS_pred = Time_Varying_Vector_Autoregressive.TV_VAR_OKS(r, y_t[:ind_1], t[:ind_1])
        OKS_pred_temp.append(OKS_pred.OKS_main(2))
        VAR_pred = Time_Varying_Vector_Autoregressive.TV_VAR_trad(r, y_t[:ind_1], t[:ind_1])
        VAR_pred_temp.append(VAR_pred.VAR_main(2))
    GAM_pred_p2_3.append(GAM_pred_temp)
    OKS_pred_p2_3.append(OKS_pred_temp)
    VAR_pred_p2_3.append(VAR_pred_temp)
    y_true_p2_3.append(y_t[ind_1:])
print(MAE_statis(GAM_pred_p2_3,y_true_p2_3,r=2, H=2))
print(MAE_statis(OKS_pred_p2_3,y_true_p2_3,r=2, H=2))
print(MAE_statis(VAR_pred_p2_3,y_true_p2_3,r=2, H=2))

# Scenario 3
x0 = np.array([0,0])
t = np.arange(0, 1, 0.005)
ind_1 = -2
r_max = 3
p = 2
GAM_pred_p2, VAR_pred_p2, OKS_pred_p2, y_true_p2 = [], [], [], []
for i in tqdm(range(100)):
    A1_fixed = np.random.uniform(0,1,size=(2,2))
    A1_fixed = A1_fixed/np.sum(A1_fixed,axis=1).reshape(-1,1)
    A2_fixed = np.random.uniform(0,1,size=(2,2))
    A2_fixed = A2_fixed/np.sum(A2_fixed,axis=1).reshape(-1,1)
    GAM_pred_temp, OKS_pred_temp, VAR_pred_temp = [], [], []
    y_t = y_t_generate2(x0, t, p, A1_fixed, A2_fixed, A_type='p2')
    for r in range(1, r_max):
        GAM_pred = Time_Varying_Vector_Autoregressive.TV_VAR_GAM(r, y_t[:ind_1], t[:ind_1])
        GAM_pred_temp.append(GAM_pred.GAM_main(2))
        OKS_pred = Time_Varying_Vector_Autoregressive.TV_VAR_OKS(r, y_t[:ind_1], t[:ind_1])
        OKS_pred_temp.append(OKS_pred.OKS_main(2))
        VAR_pred = Time_Varying_Vector_Autoregressive.TV_VAR_trad(r, y_t[:ind_1], t[:ind_1])
        VAR_pred_temp.append(VAR_pred.VAR_main(2))
    GAM_pred_p2.append(GAM_pred_temp)
    OKS_pred_p2.append(OKS_pred_temp)
    VAR_pred_p2.append(VAR_pred_temp)
    y_true_p2.append(y_t[ind_1:])
print(MAE_statis(GAM_pred_p2,y_true_p2,r=2, H=2))
print(MAE_statis(OKS_pred_p2,y_true_p2,r=2, H=2))
print(MAE_statis(VAR_pred_p2,y_true_p2,r=2, H=2))

# Scenario 4
x0 = np.array([0,0,0])
t = np.arange(0, 1, 0.005)
ind_1 = -2
r_max = 3
p=3
GAM_pred_p2_3, VAR_pred_p2_3, OKS_pred_p2_3, y_true_p2_3 = [], [], [], []
for i in tqdm(range(100)):
    A1_fixed = np.random.uniform(0,1,size=(3,3))
    A1_fixed = A1_fixed/np.sum(A1_fixed,axis=1).reshape(-1,1)
    A2_fixed = np.random.uniform(0,1,size=(3,3))
    A2_fixed = A2_fixed/np.sum(A2_fixed,axis=1).reshape(-1,1)
    y_t = y_t_generate2(x0, t, p, A1_fixed, A2_fixed, A_type='p3_1')
    GAM_pred_temp, OKS_pred_temp, VAR_pred_temp = [], [], []
    for r in range(1, r_max):
        GAM_pred = Time_Varying_Vector_Autoregressive.TV_VAR_GAM(r, y_t[:ind_1], t[:ind_1])
        GAM_pred_temp.append(GAM_pred.GAM_main(2))
        OKS_pred = Time_Varying_Vector_Autoregressive.TV_VAR_OKS(r, y_t[:ind_1], t[:ind_1])
        OKS_pred_temp.append(OKS_pred.OKS_main(2))
        VAR_pred = Time_Varying_Vector_Autoregressive.TV_VAR_trad(r, y_t[:ind_1], t[:ind_1])
        VAR_pred_temp.append(VAR_pred.VAR_main(2))
    GAM_pred_p2_3.append(GAM_pred_temp)
    OKS_pred_p2_3.append(OKS_pred_temp)
    VAR_pred_p2_3.append(VAR_pred_temp)
    y_true_p2_3.append(y_t[ind_1:])
print(MAE_statis(GAM_pred_p2_3,y_true_p2_3,r=2, H=2))
print(MAE_statis(OKS_pred_p2_3,y_true_p2_3,r=2, H=2))
print(MAE_statis(VAR_pred_p2_3,y_true_p2_3,r=2, H=2))