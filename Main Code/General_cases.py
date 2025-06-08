import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from Multiply_updates import MultiplicativeUpdates

### Define the traditional PGD
def simplex_lasso_project(x, z):
    x_abs = np.abs(x)
    if np.sum(x_abs) <= z:
        return x
    else:
        u = np.sort(x_abs)[::-1]
        u_cum = np.cumsum(u)
        rho = np.where(u > (u_cum - z) / (np.arange(1, len(x)+1)))[0][-1]
        theta = (u_cum[rho] - z) / (rho + 1)
        return np.sign(x) * np.maximum(x_abs - theta, 0)
    
def pgd_lasso(x, c, A, b, run_am, linear_rate = 0.01, lambda_pgd=0, lambda_linear_rate=0.1):
    obj = [x.reshape(-1,)]
    for k in range(run_am):
        grad = A @ x + b
        x = x - linear_rate * grad
        x = simplex_lasso_project(x, c)
        lambda_pgd = lambda_pgd + lambda_linear_rate * (np.sum(x) - c)
        obj.append(x.reshape(-1,))
    return x

def disc_project(x, center, r):
    dist_x_cent = x - center
    norm_sq = np.dot(dist_x_cent, dist_x_cent)
    if norm_sq <= r:
        return x
    else:
        return center + dist_x_cent * np.sqrt(r) / np.sqrt(norm_sq)

def pgd_disc(x, center, r, A, b, run_am, linear_rate = 0.01, lambda_ex2=0, lambda_linear_rate=0.1):
    obj = [x.reshape(-1,)]
    for k in range(run_am):
        grad = A @ x + b + 2*lambda_ex2*(x-center)
        x = x - linear_rate * grad
        x = disc_project(x, center, r)
        lambda_ex2 = lambda_ex2 + lambda_linear_rate * (np.dot(x-center,x-center) - r)
        lambda_ex2 = np.max(lambda_ex2, 0)
        obj.append(x.reshape(-1,))
    return x

def pgd_rectangle(x, low_bounds, upon_bounds, A, b, run_am, linear_rate = 0.01):
    obj = [x.reshape(-1,)]
    for k in range(run_am):
        grad = A @ x+ b
        x = x - linear_rate * grad
        x = np.minimum(np.maximum(x, low_bounds), upon_bounds)
        obj.append(x.reshape(-1,))
    return x

# General Case 1
def objective_function(x, A, b):
    x = x.reshape(-1,1)
    return (np.transpose(x).dot(A).dot(x)/2 + b.reshape(1,-1).dot(x)).item()

def constant_shift(A, b, shift_cons, orignal_center, r, run_am):
    b_new = b - shift_cons*(A.dot(np.ones(shape=(A.shape[0],1)))).reshape(-1,)
    center_new = orignal_center + shift_cons
    MU_th = MultiplicativeUpdates(A, b_new, domain='disc', r=r, center=center_new, run_am=run_am)
    MU_out = MU_th.MU_main()
    return MU_out - shift_cons

# General Case 2 and 3
def sig_matrix(low_bounds, upon_bounds):
    k = len(low_bounds)
    all_bin_one = []
    bounds_array = np.array([low_bounds, upon_bounds])
    for i in range(2**k):
        all_bin_one.append(bin(i)[2:].zfill(k))
    sig_out, block_list = [], []
    for sig in all_bin_one:
        sig_one_temp = np.zeros(k)
        block_one_low = np.zeros(shape=(k)).tolist()
        block_one_temp = []
        for s in range(len(sig)):
            sig_one_temp[s] = np.array(sig[s]).astype('int')
            block_one_temp.append(bounds_array[int(sig[s]),s])
        sig_one_temp = np.sign(sig_one_temp-0.5)
        sig_out.append(np.diag(sig_one_temp))
        block_list.append(np.asarray([block_one_low, block_one_temp]))
    return sig_out, np.abs(block_list)

def split_flip_one(sig_one, A, b, c, run_am):
    A_siged = sig_one.dot(A).dot(sig_one)
    b_siged = b.reshape(1,-1).dot(sig_one).reshape(-1,)

    MU_th = MultiplicativeUpdates(A_siged, b_siged, domain='linear', linear_C=c, run_am=run_am)
    MU_out = MU_th.MU_main()
    return sig_one.dot(MU_out.reshape(-1,1))

def split_flip_main(A, b, c, run_am):
    ds = A.shape[1]
    sig,_ = sig_matrix(-np.ones(shape=(ds,)), np.ones(shape=(ds,)))
    new_method, new_method_value= [], []
    for i in range(len(sig)):
        new_method_temp = split_flip_one(sig[i], A, b, c, run_am)
        new_method_value.append(objective_function(new_method_temp, A, b))
        new_method.append(new_method_temp)
    return new_method[np.argmin(new_method_value)]

def split_flip_one_rect(sig_one, A, b, low_bounds, upon_bounds, run_am):
    A_siged = sig_one.dot(A).dot(sig_one)
    b_siged = b.reshape(1,-1).dot(sig_one).reshape(-1,)

    MU_th = MultiplicativeUpdates(A_siged, b_siged, domain='rectangle', L_bounds=low_bounds, U_bounds=upon_bounds, run_am=run_am)
    MU_out = MU_th.MU_main()
    return sig_one.dot(MU_out.reshape(-1,1))

def split_flip_main_rect(low_bounds, upon_bounds, A, b, run_am):
    ds = A.shape[1]
    sig, bounds_list = sig_matrix(low_bounds, upon_bounds)
    new_method, new_method_value= [], []

    for i in range(len(sig)):
        bounds = bounds_list[i]
        new_method_temp = split_flip_one_rect(sig[i], A, b, bounds[0].reshape(-1,1), bounds[1].reshape(-1,1), run_am)

        new_method_value.append(objective_function(new_method_temp, A, b))
        new_method.append(new_method_temp)
    return new_method[np.argmin(new_method_value)]

# Define the visualization function
def exa_process(ex_res):
    tar, new_m, trad_m = ex_res
    return {'MU': np.around(mean_absolute_error(np.array(tar), np.array(new_m)), 7),
            'Project Gradient Descent': np.around(mean_absolute_error(np.array(tar), np.array(trad_m)),7)}

def exa_process_Lasso_ridge(ex_res, Lasso_check= False, Ridge_check = False):
    if Lasso_check:
        tar, new_m_lasso, Lasso_m = ex_res
        return {'MU (Lasso domain)': np.around(mean_absolute_error(np.array(tar), np.array(new_m_lasso)), 7),
                'Lasso Regression': np.around(mean_absolute_error(np.array(tar), np.array(Lasso_m)),7)}
    elif Ridge_check:
        tar, new_m_ridge, ridge_m = ex_res
        return {'MU (Ridge domain)': np.around(mean_absolute_error(np.array(tar), np.array(new_m_ridge)), 7),
                'Ridge Regression': np.around(mean_absolute_error(np.array(tar), np.array(ridge_m)),7)}

def dict_process(*dicts):
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                if not isinstance(merged_dict[key], list):
                    merged_dict[key] = [merged_dict[key]]
                merged_dict[key].append(value)
            else:
                merged_dict[key] = value
    return merged_dict

# Define the main test function
def run_main(ds, run_am=1000):
    new_m_ex1, trad_m_ex1, target_ex1 = [], [], []
    new_m_ex2, trad_m_ex2, target_ex2 = [], [], []
    new_m_ex3, trad_m_ex3_lasso, target_ex3 = [], [], []
    new_m_ex4, trad_m_ex4_ridge, target_ex4  = [], [], []
    for i in tqdm(range(300)):
        # Fix the random, seed from 0 to 500
        np.random.seed(i)
        # Ex1
        r = 2
        orignal_center = np.zeros(shape=(ds,))
        shift_cons = 10
        x_target = orignal_center + np.random.uniform(-1.5,1.5,size=(ds,))
        k = 1
        while LA.norm(x_target - orignal_center)>r:
            x_target = orignal_center + np.random.uniform(-1.5/k,1.5/k,size=(ds,))
            k = k+0.2
        temp = np.random.normal(0,1, size=(ds, ds))
        A = np.dot(temp, np.transpose(temp))
        b = -A.dot(x_target)
        target_ex1.append(x_target)
        New_m_1 = constant_shift(A, b, shift_cons, orignal_center, r, run_am)
        new_m_ex1.append(New_m_1.reshape(-1,))
        Trad_m_1 = pgd_disc(orignal_center, orignal_center, r, A, b, run_am)
        trad_m_ex1.append(Trad_m_1.reshape(-1,))
        
        # Ex2
        low_bounds = np.random.randint(-5, 5, size=(ds,))
        upon_bounds = low_bounds + np.random.randint(2,10, size=(ds,))
        # Both low_bounds and upon_bounds don't have inifity value at begining
        x_target = np.random.uniform(low_bounds, upon_bounds, size=(ds,))
        # Generate matrix A
        Q, _ = np.linalg.qr(np.random.randn(ds, ds))
        A = Q.dot(np.diag(np.abs(np.random.randn(ds,)))).dot(np.transpose(Q))
        b = -A.dot(x_target)
        target_ex2.append(x_target)
        low_bounds = low_bounds.astype('float')
        trad_ini_x = np.mean([low_bounds, upon_bounds], axis=0)
        low_bounds[np.random.choice(range(ds), size=(np.random.choice(range(1,ds))), replace=False)] = -np.inf    

        New_m_2 = split_flip_main_rect(low_bounds, upon_bounds, A, b, run_am)
        new_m_ex2.append(New_m_2.reshape(-1,))
        
        Trad_m_2 = pgd_rectangle(trad_ini_x, low_bounds, upon_bounds, A, b, run_am)
        trad_m_ex2.append(Trad_m_2.reshape(-1,))

        # trad_m_ex3_ridge.append(-ridge_regression(A, b))
        # Ex3
        x_target = np.random.uniform(-1,1,size=(ds,))
        c = 1
        j = 1
        while np.sum(np.abs(x_target))>c:
            x_target = np.random.uniform(-1/j,1/j,size=(ds,))
            j = j+0.5
        target_ex3.append(x_target)
        Q, _ = np.linalg.qr(np.random.randn(ds, ds))
        A = Q.dot(np.diag(np.abs(np.random.randn(ds,)))).dot(np.transpose(Q))
        b = -A.dot(x_target)
        new_m_ex3.append(split_flip_main(A, b, c, run_am).reshape(-1,))
        trad_m_ex3_lasso.append(pgd_lasso(np.zeros(shape=(ds,)), c, A, b, run_am).reshape(-1,))

        # Ex4
        r = 1
        orignal_center = np.zeros(shape=(ds,))
        shift_cons = 3
        x_target = np.random.uniform(-1,1,size=(ds,))
        k = 1
        while LA.norm(x_target)>r:
            x_target = orignal_center + np.random.uniform(-1/k,1/k,size=(ds,))
            k = k+0.2
        target_ex4.append(x_target)

        temp = np.random.normal(0,1, size=(ds, ds))
        A = np.dot(temp, np.transpose(temp))
        b = -A.dot(x_target)
        new_m_ex4.append(constant_shift(A, b, shift_cons, orignal_center, r, run_am).reshape(-1,))
        
        trad_m_ex4_ridge.append(pgd_disc(orignal_center, orignal_center, r, A, b, run_am).reshape(-1,))

        
    return [np.array(target_ex1), np.array(new_m_ex1), np.array(trad_m_ex1)],\
           [np.array(target_ex2), np.array(new_m_ex2), np.array(trad_m_ex2)],\
           [np.array(target_ex3), np.array(new_m_ex3), np.array(trad_m_ex3_lasso)],\
           [np.array(target_ex4), np.array(new_m_ex4), np.array(trad_m_ex4_ridge)]

ex1_ds3, ex2_ds3, ex3_ds3, ex4_ds3 = run_main(3, run_am=800)
ex1_ds5, ex2_ds5, ex3_ds5, ex4_ds5 = run_main(5, run_am=800)

MAE_ex1_ds3 = exa_process(ex1_ds3)
MAE_ex2_ds3 = exa_process(ex2_ds3)
dict_ex12_ds3 = dict_process(MAE_ex1_ds3, MAE_ex2_ds3)

MAE_ex1_ds5 = exa_process(ex1_ds5)
MAE_ex2_ds5 = exa_process(ex2_ds5)
dict_ex12_ds5 = dict_process(MAE_ex1_ds5, MAE_ex2_ds5)

MAE_ex3_ds3 = exa_process_Lasso_ridge(ex3_ds3, Lasso_check=True)
MAE_ex4_ds3 = exa_process_Lasso_ridge(ex4_ds3, Ridge_check=True)
dict_ds4 = dict_process(MAE_ex3_ds3, MAE_ex4_ds3)

MAE_ex3_ds5 = exa_process_Lasso_ridge(ex3_ds5, Lasso_check=True)
MAE_ex4_ds5 = exa_process_Lasso_ridge(ex4_ds5, Ridge_check=True)
dict_ds5 = dict_process(MAE_ex3_ds5, MAE_ex4_ds5)

# Draw the Figure
x_ticks = ("C 1, d 3", "C 2, d 3", "C 1, d 5", "C 2, d 5")
width = 0.25
fig, axs = plt.subplots(layout='constrained', nrows=1, ncols=2, figsize=(8, 4))
axs[0].bar(np.array([0, 1]), dict_ex12_ds3['MU'][:2], width, color = 'slateblue', label='MU')
axs[0].bar(np.array([0, 1]) + 0.25,  dict_ex12_ds3['Project Gradient Descent'][:2], width, color = 'green', label='PGDL')
axs[0].bar(np.array([2, 3]), dict_ex12_ds5['MU'][:2], width, color = 'slateblue')
axs[0].bar(np.array([2, 3]) + 0.25,  dict_ex12_ds5['Project Gradient Descent'][:2], width, color = 'green')

axs[0].set_ylabel('MAE of predict and target')
axs[0].set_title('General Case 1 & 2')
axs[0].legend()
axs[0].set_xticks(np.arange(len(x_ticks)) + width, x_ticks)
axs[0].grid()

width2 = 0.17
axs[1].bar(np.array([0]), dict_ds4['MU (Lasso domain)'], width2, color = 'slateblue', label='MU (Lasso\n domain)')
axs[1].bar(np.array([0]) + width2,  dict_ds4['Lasso Regression'], width2, color = 'pink', label='Lasso')
axs[1].bar(np.array([0]) + 2*width2,  dict_ds4['MU (Ridge domain)'], width2, color = 'gray', label='MU (Ridge\n domain)')
axs[1].bar(np.array([0]) + 3*width2,  dict_ds4['Ridge Regression'], width2, color = 'yellow', label='Ridge')

axs[1].bar(np.array([1]), dict_ds5['MU (Lasso domain)'], width2, color = 'slateblue')
axs[1].bar(np.array([1]) + width2,  dict_ds5['Lasso Regression'], width2, color = 'pink')
axs[1].bar(np.array([1]) + 2*width2,  dict_ds5['MU (Ridge domain)'], width2, color = 'gray')
axs[1].bar(np.array([1]) + 3*width2,  dict_ds5['Ridge Regression'], width2, color = 'yellow')
axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
axs[1].set_yscale('log')
axs[1].set_title('Lasso & Ridge')
axs[1].set_xticks(np.array([0.28, 1.28]), ("d 3", "d 5"))
axs[1].grid()

fig.suptitle('MAE of Multiplicative Update vs Project Gradient Descent Lagrangian', fontsize=14, fontweight="bold", y=1.05)
plt.show()