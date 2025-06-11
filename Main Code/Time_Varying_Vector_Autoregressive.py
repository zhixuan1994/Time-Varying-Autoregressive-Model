import numpy as np
import numpy.linalg as LA
import pandas as pd
from typing import Union, List
from scipy.optimize import minimize, NonlinearConstraint
from tqdm import tqdm

class TV_VAR_GAM:
    def __init__(self, lag_r: int, series: Union[List, np.ndarray, pd.Series], 
                 t_list: Union[List, np.ndarray, pd.Series]):
        self.lag_r = lag_r
        self.series_data = np.array(series)
        self.t_list = t_list
    
    def all_basis_func(self, t, k_order=3):
        t = 2*(t-0.5)
        t = t.reshape(-1,1)
        # Linear with order
        linear_list = t
        for linear_order in range(2, k_order):
            temp = (t)**linear_order
            linear_list = np.concatenate([linear_list, temp], axis=1)
        # Tanh
        tanh = (np.exp(5*t)-np.exp(-5*t)) / (np.exp(5*t)+np.exp(-5*t))
        # Gaussian
        gaus = np.exp(-5*t**2)
        # Trad Exp
        trad_exp = np.exp(t/5)
        # Trad Cos
        trad_cos = np.cos(t/5)
        return np.concatenate([linear_list, gaus, trad_exp, trad_cos], axis=1)

    def TV_VAR_lagged(self, linear_max_order=3):
        t_train = []
        for i in range(self.lag_r):
            t_train.append(self.t_list[i:-self.lag_r+i])
        t_train = np.transpose(np.array(t_train))
        t_train_processed = self.all_basis_func(t_train[:, -1] + self.t_list[1]-self.t_list[0])
        len_basis_func = len(t_train_processed)
        for i in range(t_train.shape[1]):
            t_processed_temp = self.all_basis_func(t_train[:,i], k_order=linear_max_order)
            t_processed_kroned = np.array(np.kron(self.series_data[i], t_processed_temp[0])).reshape(1,-1)
            for j in range(1, t_train.shape[0]):
                t_kroned_temp = np.kron(self.series_data[i+j], t_processed_temp[j]).reshape(1,-1)
                t_processed_kroned = np.concatenate([t_processed_kroned, t_kroned_temp], axis=0)
            t_train_processed = np.concatenate([t_train_processed, t_processed_kroned], axis=1)
        return t_train_processed, len_basis_func
    
    def linear_minimize_obj(self, x, A, y):
        x = x.reshape(-1,1)
        return LA.norm(A.dot(x) - y)

    def GAM_constrain(self, x, p):
        r = int((len(x) - 1)/p)
        H = np.zeros(shape=(r, r*p + p))
        for i in range(r):
            H[i, p+i*p: p+p*(i+1)] = 1
        return np.sum(np.abs(H.dot(x)))
    
    def TV_VAR_GAM_one_index(self, dimen_index):
        TV_VAR_data, len_basis_func = self.TV_VAR_lagged()
        TV_VAR_target = self.series_data[self.lag_r:, dimen_index]
        TV_VAR_constraint = NonlinearConstraint(lambda x: self.GAM_constrain(x, len_basis_func), 0, 1)
        TV_VAR_x = minimize(self.linear_minimize_obj, np.ones(shape=TV_VAR_data.shape[1]), method='trust-constr',
                        args = (TV_VAR_data, TV_VAR_target.reshape(-1,1)), constraints=[TV_VAR_constraint]).x
        return TV_VAR_x
    
    def GAM_predict(self, TV_VAR_x_list, predict_data, t_list_r, t_h, predict_step):
        if len(TV_VAR_x_list) < self.series_data.shape[0]:
            print('No enough input')
        TV_VAR_data = self.all_basis_func(t_list_r[-1]+t_h).reshape(-1,)
        for j in range(self.lag_r):
            t_processed_temp = self.all_basis_func(t_list_r[j])
            t_kroned_temp = np.kron(predict_data[j], t_processed_temp).reshape(-1,)
            TV_VAR_data = np.concatenate([TV_VAR_data, t_kroned_temp])
        pred_res = []
        for h in range(1, predict_step+1):
            res_temp = []
            for i in range(self.series_data.shape[0]):
                res_temp.append((TV_VAR_data.dot(TV_VAR_x_list[i].reshape(-1,1))))
            res_temp = np.array(res_temp).reshape(1,-1)
            pred_res.append(res_temp.reshape(-1,))
            predict_data = np.concatenate([predict_data[1:], res_temp], axis=0)
            TV_VAR_data = self.all_basis_func(t_list_r[-1]+t_h*(h+1)).reshape(-1,)
            for j in range(self.lag_r):
                t_processed_temp = self.all_basis_func(t_list_r[j] + t_h*h)
                t_kroned_temp = np.kron(predict_data[j], t_processed_temp).reshape(-1,)
                TV_VAR_data = np.concatenate([TV_VAR_data, t_kroned_temp])
        return np.array(pred_res)
    
    def GAM_main(self, predict_step):
        GAM_pred_all = []
        t_h = self.t_list[1] - self.t_list[0]
        for r in range(1,self.lag_r):
            GAM_r = []
            for i in range(len(self.series_data)):
                GAM_r.append(self.TV_VAR_GAM_one_index(i))
            GAM_pred = self.GAM_predict(GAM_r, self.series_data[-r:], self.t_list[-r:], t_h, predict_step)
            GAM_pred_all.append(GAM_pred)
        return GAM_pred_all
    
class TV_VAR_OKS:
    def __init__(self, lag_r: int, series: Union[List, np.ndarray, pd.Series], 
                 t_list: Union[List, np.ndarray, pd.Series]):
        self.lag_r = lag_r
        self.series_data = np.array(series)
        self.t_list = t_list

    def kernel_weight(self, t, t_star, h):
        return np.exp(-(t-t_star)**2 / (2*h**2)) / np.sqrt(2*np.pi*h**2)

    def A_t_star_main(self, t_star):
        A = np.ones(shape=(t_star - self.lag_r - 1, 1))
        Y_temp = []
        for i in range(t_star - self.lag_r - 1):
            A_temp = self.series_data[i:i+self.lag_r].reshape(-1,)
            Y_temp.append(A_temp)
        return np.concatenate([A, np.array(Y_temp)], axis=1)

    def OKS_constraint(self, x):
        H = np.array([0])
        H = np.concatenate([H, np.ones(shape=(len(x)-1,))])
        return H.dot(np.abs(x))

    def one_side_KS_t_star_y_ind(self, t_star, y_ind, h=0.5):
        A_t_star = self.A_t_star_main(t_star)
        t_list = np.arange(self.lag_r+1, t_star,1)
        Y_list_ind = self.series_data[self.lag_r+1:t_star, y_ind].reshape(-1,1)
        W_t_star = self.kernel_weight(t_list, t_star, h=h).reshape(-1,1)
        A_t_star = W_t_star*A_t_star
        Y_list_ind = W_t_star*Y_list_ind
        TVAR_constraint = NonlinearConstraint(lambda x: self.OKS_constraint(x), 0, 1)
        TV_VAR_x = minimize(self.linear_minimize_obj, np.ones(shape=A_t_star.shape[1])*0.5, method='trust-constr',
                        args = (A_t_star, Y_list_ind.reshape(-1,1)), constraints=[TVAR_constraint]).x
        return TV_VAR_x  
    
    def one_side_KS_CV_main(self, t_star, y_ind, h_list = np.linspace(5, 26, 21)):    
        last_mae = []
        y_temp = np.concatenate([[1], self.series_data[-self.lag_r-1:-1].reshape(-1)])
        for h in h_list:
            last_one_coef = self.one_side_KS_t_star_y_ind(t_star-1, y_ind, h=h)
            last_mae.append(np.abs(y_temp @ last_one_coef - self.series_data[-1, y_ind]))
        h = h_list[np.argmin(last_mae)]
        return self.one_side_KS_t_star_y_ind(t_star, y_ind, h=h)
    
    def one_side_KS_predict(Y, beta_t_star_ma, predict_step):
        Y_pred_data = Y
        Y_pred_res, Y_pred_temp = [], []
        for k in range(Y.shape[1]):
            Y_pred_data_temp = np.concatenate([[1], Y_pred_data.reshape(-1)])
            Y_pred_temp.append(Y_pred_data_temp @ beta_t_star_ma[k])
        Y_pred_temp = np.array(Y_pred_temp)
        Y_pred_data = np.concatenate([Y_pred_data, Y_pred_temp.reshape(1,-1)], axis=0)[1:]
        Y_pred_res.append(Y_pred_temp)

        for _ in range(1, predict_step):
            Y_pred_temp = []
            for k in range(Y.shape[1]):
                Y_pred_data_temp = np.concatenate([[1], Y_pred_data.reshape(-1)])
                Y_pred_temp.append(Y_pred_data_temp @ beta_t_star_ma[k])
            Y_pred_temp = np.array(Y_pred_temp)
            Y_pred_data = np.concatenate([Y_pred_data, Y_pred_temp.reshape(1,-1)], axis=0)[1:]
            Y_pred_res.append(Y_pred_temp)
        return Y_pred_res
    
    def OKS_pred_H_offline(self, predict_step):
        OKS_pred_all = []
        for r in range(1,self.lag_r):
            OKS_r = []
            for i in range(len(self.series_data)):
                OKS_r.append(self.one_side_KS_CV_main(self.series_data, r, len(self.t_list), i))
            OKS_pred = self.one_side_KS_predict(self.series_data[-r:], OKS_r, predict_step)
            OKS_pred_all.append(OKS_pred)
        return np.array(OKS_pred_all)

class TV_VAR_trad:
    def __init__(self, lag_r: int, series: Union[List, np.ndarray, pd.Series], 
                 t_list: Union[List, np.ndarray, pd.Series]):
        self.lag_r = lag_r
        self.series_data = np.array(series)
        self.t_list = t_list
        self.VAR_shape_0 = len(self.series_data)

    def estimate_var_coefficients(self):
        VAR_shape_0 = self.VAR_shape_0
        X = np.hstack([self.series_data[VAR_shape_0 - i - 1:self.series_data.shape[0] - i - 1] for i in range(VAR_shape_0)])
        Y_target = self.series_data[VAR_shape_0:]
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        B_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ Y_target)
        intercept = B_hat[0]
        A_flat = B_hat[1:].T
        A_list = [A_flat[:, i*self.series_data.shape[1]:(i+1)*self.series_data.shape[1]] for i in range(VAR_shape_0)]
        return intercept, A_list

    def forecast_var_p(Y_hist, A_list, intercept, predict_step):
        p = len(A_list)
        preds = []
        Y_current = Y_hist[-p:].tolist()
        for _ in range(predict_step):
            y_pred = intercept.copy()
            for i in range(p):
                y_pred += A_list[i] @ Y_current[-i-1]
            preds.append(y_pred)
            Y_current.append(y_pred)
            Y_current.pop(0)
        return np.array(preds)
    
    def VAR_pred_H_offline(self, predict_step):
        VAR_pred_all = []
        for r in range(1, self.lag_r):
            VAR_intercept, VAR_A_list = self.estimate_var_coefficients(self.series_data, r)
            VAR_pred_all.append(self.forecast_var_p(self.series_data, VAR_A_list, VAR_intercept, predict_step))
        return VAR_pred_all
