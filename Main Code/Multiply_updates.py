import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve
from sklearn.metrics import mean_absolute_error

class MultiplicativeUpdates:
    def __init__(self, A, b, domain: str, L_bounds: np.array=(), U_bounds: np.array=(), center: np.array=(), 
                 r: float=1, eta: np.array=(), linear_C: float=1, pol_b1: float=0, pol_b2: float=4, run_am: int=200):
        self.A = A
        self.b = b
        self.domain = domain
        self.L_bounds = L_bounds
        self.U_bounds = U_bounds
        self.center = center
        self.linear_C = linear_C
        self.pol_b1 = pol_b1
        self.pol_b2 = pol_b2
        self.run_am = run_am
        self.r = r
        self.eta = eta
        if self.domain not in ['rectangle','disc', 'ellipse','linear']:
            raise ValueError('Only support domain of: rectangle, disc, ellipse, linear')

    def M_matrix(self, A, b, v):
        A_p = np.where(A >= 0, A, 0)
        A_n = np.abs(np.where(A < 0, A, 0))
        a_vec = A_p.dot(v)
        c_vec = A_n.dot(v)
        a_vec = a_vec.reshape(-1,)
        # Avoid 0 in vector a
        for i in range(len(a_vec)):
            if np.abs(a_vec[i]) < 1e-10:
                a_vec[i] = 1e-10
        c_vec = c_vec.reshape(-1,)
        b_vec = b.reshape(-1,)
        temp = b_vec**2 + 4*a_vec*c_vec
        # Avoid negative Sqrt value
        m = -b_vec + np.sqrt(np.max(np.concatenate([0.0001*np.ones(shape=(len(temp),1)),temp.reshape(-1,1)], axis=1), axis=1))
        m = m / (2*a_vec)
        return np.diag(m)

    # Rectangle
    def a_rectangle(self, L, U, Mv, v):
        a_list = []
        for i in range(len(Mv)):
            a_list.append((L[i] - Mv[i]) / (v[i] - Mv[i]))
            a_list.append((U[i] - Mv[i]) / (v[i] - Mv[i]))
        a_list = np.array(a_list).reshape(-1,)
        a_list = a_list[a_list <= 1]
        if len(a_list) == 0:
            return 0
        else:
            a_star_temp = np.max(a_list)
            if a_star_temp < 0:
                return 0
            else:
                return a_star_temp

    def v_rectangle(self, M, v, L, U):
        Mv = np.dot(M, v)
        upon_check = (Mv - U) > -1e-10
        low_check = (Mv - L) < 1e-10
        if (True in upon_check) or (True in low_check):
            a = self.a_rectangle(L, U, Mv, v)
            v_temp = ((a*v + (1-a)*Mv) + v)/2
            return v_temp
        else:
            return Mv

    def main_rectangle(self, D_low, D_upon, A, b, v):
        D_low = D_low.reshape(-1,1)
        D_upon = D_upon.reshape(-1,1)
        b = b.reshape(-1,1)
        v = v.reshape(-1,1)
        M = self.M_matrix(A, b, v)
        return self.v_rectangle(M, v, D_low, D_upon)

    # Disc
    def a_disc(self, Mv,v, r, center):
        v_Mv_center = np.dot((v-Mv).reshape(1,-1), Mv-center)
        upon = -v_Mv_center - np.sqrt(v_Mv_center**2 - LA.norm(v-Mv)**2 * (LA.norm(Mv-center)**2 - r**2))
        down = LA.norm(v-Mv)**2
        return upon / down

    def v_disc(self, M, v, r, center):
        Mv = np.dot(M, v)
        check_temp = LA.norm(Mv-center)
        if check_temp > r:
            a = self.a_disc(Mv, v, r, center)
            out = (a*v + (1-a)*Mv + v)/2
            return out
        else:
            return Mv

    def main_disc(self, r, A, b, v, center):
        b = b.reshape(-1,1)
        v = v.reshape(-1,1)
        M = self.M_matrix(A, b, v)
        center = center.reshape(-1,1)
        return self.v_disc(M, v, r, center)

    # Convex linear functions
    def a_linear(self, M, v, c):
        Mv = np.dot(M, v)
        if np.any(Mv < 0):
            Mv_below_0 = Mv<0
            Mv_upon_c = Mv>c
            Mv_list = np.concatenate([Mv[Mv_below_0], Mv[Mv_upon_c]], axis=0)
            v_list = np.concatenate([v[Mv_below_0], v[Mv_upon_c]], axis=0)
            Mv_0 = np.abs(Mv_list).reshape(-1,1)
            Mv_c = np.abs(Mv_list-c).reshape(-1,1)
            Mv_abs_min = np.min(np.concatenate([Mv_0, Mv_c], axis=1),axis=1)
            k_star_temp = np.argmin(Mv_abs_min)
            if Mv_abs_min[k_star_temp] == Mv_0[k_star_temp]:
                b_star = 0
            else:
                b_star = c
            v_out = (Mv_list[k_star_temp] - b_star)/(Mv_list[k_star_temp] - v_list[k_star_temp])
            return v_out
        else:
            return (c-np.sum(Mv))/(np.sum(v) - np.sum(Mv))

    def v_linear(self, M, v, c):
        Mv = np.dot(M, v)
        check_temp = np.sum(Mv)
        if check_temp - c > 1e-10 or np.any(Mv < -1e-10):
            a= self.a_linear(M, v, c)
            return (a*v + (1-a)*Mv + v)/2
        else:
            return Mv

    def main_linear(self, A, b, v, c):
        b = b.reshape(-1,1)
        v = v.reshape(-1,1)
        M = self.M_matrix(A, b, v)
        return self.v_linear(M, v, c)
    
    def MU_main(self):
        A = self.A
        b = self.b
        run_am = self.run_am
        if self.domain == 'rectangle':
            U_bounds_inf_mask = np.isinf(self.U_bounds)
            U_bounds_temp = self.U_bounds.copy()
            U_bounds_temp[U_bounds_inf_mask] = self.L_bounds[U_bounds_inf_mask] + 10
            initial_val = np.mean([self.L_bounds, U_bounds_temp],axis=0)
            MU_out = self.main_rectangle(self.L_bounds, self.U_bounds, A, b, initial_val).reshape(-1,)
            for i in range(run_am):
                MU_out = self.main_rectangle(self.L_bounds, self.U_bounds, A, b, MU_out).reshape(-1,)

        elif self.domain == 'disc':
            MU_out = self.main_disc(self.r, A, b, self.center, self.center)
            for i in range(run_am):
                MU_out = self.main_disc(self.r, A, b, MU_out, self.center)

        elif self.domain == 'ellipse':
            eta = self.eta
            A_trans = np.diag(eta).dot(A).dot(np.diag(eta))
            b_trans = np.diag(eta).dot(b.reshape(-1,1))
            center_trans = self.center/eta
            initial_trans = self.center/eta
            MU_out = self.main_disc(self.r, A_trans, b_trans, initial_trans, center_trans)
            for i in range(run_am):
                MU_out = self.main_disc(self.r, A_trans, b_trans, MU_out, center_trans)
            MU_out = MU_out.reshape(-1,)*eta.reshape(-1,)
            
        elif self.domain == 'linear':
            initial_val = np.ones(shape=(A.shape[0],))*self.linear_C/A.shape[0]*0.5
            MU_out = self.main_linear(A, b, initial_val, self.linear_C)
            for i in range(run_am):
                MU_out = self.main_linear(A, b, MU_out, self.linear_C)
        return MU_out
