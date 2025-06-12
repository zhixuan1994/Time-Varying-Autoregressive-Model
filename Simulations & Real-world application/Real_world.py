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

TV_VAR_data = pd.read_csv('TV_VAR_real_wd_data.cvs')
TV_VAR_data['Date_time'] = pd.to_datetime(TV_VAR_data['Date_time'])
TV_VAR_data_np = TV_VAR_data.iloc[:, 1:].to_numpy()

# Predict the future with 1 step by TV-VAR(2)
y_t = TV_VAR_data_np
t = np.linspace(0, 1, len(y_t))
GAM_pred_all, OKS_pred_all, VAR_pred_all = [], [], []
for i in tqdm(range(126)):
    GAM_pred_temp = Time_Varying_Vector_Autoregressive.TV_VAR_GAM(2, y_t[:-126 + i], t[:-126 + i])
    OKS_pred_temp = Time_Varying_Vector_Autoregressive.TV_VAR_OKS(2, y_t[:-126 + i], t[:-126 + i])
    GAM_pred_all.append(GAM_pred_temp.GAM_main(1))
    OKS_pred_all.append(OKS_pred_temp.OKS_main(1))
GAM_pred_all = np.array(GAM_pred_all)
OKS_pred_all = np.array(OKS_pred_all)

# Saving the prediction results
GAM_pred_r,OKS_pred_r = [], []
for i in range(len(GAM_pred_all)):
    GAM_pred_r.append(GAM_pred_all[i].reshape(-1,))
    OKS_pred_r.append(OKS_pred_all[i].reshape(-1,))
GAM_pred_r = np.array(GAM_pred_r)
OKS_pred_r = np.array(OKS_pred_r)
GAM_pred_df = pd.DataFrame(data=GAM_pred_r, columns=['VIX', 'SP500','US-1-month'])
OKS_pred_df = pd.DataFrame(data=OKS_pred_r, columns=['VIX', 'SP500','US-1-month'])
GAM_pred_df.to_csv('GAM_rd_126.csv', index=False)
OKS_pred_df.to_csv('OKS_rd_126.csv', index=False)

# Reading and analysis the results
GAM_pred_df = pd.read_csv('GAM_rd_126.csv')
OKS_pred_df = pd.read_csv('OKS_rd_126.csv')
TV_VAR_data = pd.read_csv('TV_VAR_real_wd_data.cvs')
TV_VAR_data['Date_time'] = pd.to_datetime(TV_VAR_data['Date_time'])
y_true = TV_VAR_data.iloc[-126:]
y_true = y_true.reset_index(drop='True')

GAM_MAE, OKS_MAE = [], []
for i in range(3):
    GAM_MAE.append(mean_absolute_error(y_true.iloc[:, i+1], GAM_pred_df.iloc[:,i]))
    OKS_MAE.append(mean_absolute_error(y_true.iloc[:, i+1], OKS_pred_df.iloc[:,i]))
GAM_MAE = np.around(GAM_MAE, 3)
OKS_MAE = np.around(OKS_MAE, 3)

# Visualization
fig, axs=plt.subplots(layout='constrained', nrows=2, ncols=2, figsize=(13, 7))
axs[0,0].plot(y_true['date_column'], GAM_pred_df.iloc[:,0], label='GAM(2)', color='slateblue', linewidth=1.5)
axs[0,0].plot(y_true['date_column'], OKS_pred_df.iloc[:,0], label='OKS(2)', color='brown', linewidth=1.5)
axs[0,0].plot(y_true['date_column'], y_true.iloc[:,1], '.r', label='True value', ms=3)
axs[0,0].grid()
axs[0,0].set_title('VIX index')
axs[0,0].set_ylabel('VIX')

axs[0,1].plot(y_true['date_column'], GAM_pred_df.iloc[:,1], color='slateblue', linewidth=1.5)
axs[0,1].plot(y_true['date_column'], OKS_pred_df.iloc[:,1], color='brown', linewidth=1.5)
axs[0,1].plot(y_true['date_column'], y_true.iloc[:,2],'.r', ms=3)
axs[0,1].grid()
axs[0,1].set_title('S&P 500 index')
axs[0,1].set_ylabel('S&P 500 price')

axs[1,0].plot(y_true['date_column'], GAM_pred_df.iloc[:,2], color='slateblue', linewidth=1.5)
axs[1,0].plot(y_true['date_column'], OKS_pred_df.iloc[:,2], color='brown', linewidth=1.5)
axs[1,0].plot(y_true['date_column'], y_true.iloc[:,3], '.r', ms=3)
axs[1,0].grid()
axs[1,0].set_title('U.S. Treasury securities constant maturity (1-month)')
axs[1,0].set_ylabel('Treasury securities')

axs[1,1].bar(np.linspace(0,1,3), GAM_MAE, 0.25, color = 'slateblue')
axs[1,1].bar(np.linspace(1.5,2.5,3), OKS_MAE, 0.25, color = 'brown')
axs[1,1].set_yscale("log")
axs[1,1].set_title('MAE for all index')
axs[1,1].grid()
axs[1,1].set_xticks(np.concatenate([np.linspace(0,1,3), np.linspace(1.5,2.5,3)]), ['VIX', 'S&P500','Securities', 'VIX', 'S&P500','Securities'])
axs[1,1].set_ylabel('Mean Absolute Error')
axs[1,1].margins(y=0.2)
fig.legend(loc='upper left', bbox_to_anchor=(1, 1))