import numpy as np
from rcgp.kernels import RBFKernel
from scipy.optimize import minimize
from numpy.linalg import cholesky, solve

def imq_kernel(y, x, beta, c):
    imq = beta * (1 + ((y-x)**2)/(c**2))**(-0.5)
    gradient_log_squared = 2 * (x - y)/(c**2) * (1+(y-x)**2/(c**2))**-1
    return imq, gradient_log_squared

class GPRegressor:
    def __init__(self, mean, length_scale=1.0, rbf_variance=1.0, noise=1e-2):
        self.mean = mean
        self.length_scale = length_scale
        self.rbf_variance = rbf_variance
        self.noise = noise

    def rbf_kernel(self, X1, X2, length_scale, variance):
        """Compute the RBF kernel with variance (amplitude squared)"""
        dists = np.sum(X1**2, axis=1)[:, None] + \
                np.sum(X2**2, axis=1)[None, :] - \
                2 * X1 @ X2.T
        return variance * np.exp(-0.5 * dists / length_scale**2)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.mean_train = self.mean(X_train)
        y_centered = y_train - self.mean_train

        self.K = self.rbf_kernel(X_train, X_train, self.length_scale, self.rbf_variance)
        self.K_noise = self.K + self.noise * np.eye(len(X_train)) + 1e-6 * np.eye(len(X_train))

        self.L = cholesky(self.K_noise)
        self.alpha = solve(self.L.T, solve(self.L, y_centered))

    def predict(self, X_test):
        K_s = self.rbf_kernel(self.X_train, X_test, self.length_scale, self.rbf_variance)
        K_ss = self.rbf_kernel(X_test, X_test, self.length_scale, self.rbf_variance) + \
               1e-6 * np.eye(len(X_test))

        mu = K_s.T @ self.alpha + self.mean(X_test)  # Add mean back
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v
        return mu.reshape(-1), np.diag(cov)

    def log_marginal_likelihood(self, theta=None):
        if theta is not None:
            length_scale, noise, variance = np.exp(theta)
            K = self.rbf_kernel(self.X_train, self.X_train, length_scale, variance) + \
                noise * np.eye(len(self.X_train))
            y_centered = self.y_train - self.mean_train
        else:
            K = self.K
            y_centered = self.y_train - self.mean_train

        try:
            L = cholesky(K)
        except np.linalg.LinAlgError:
            return 1e6

        alpha = solve(L.T, solve(L, y_centered))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        return -0.5 * y_centered.T @ alpha - 0.5 * log_det - \
               0.5 * len(self.X_train) * np.log(2 * np.pi)

    def optimize_mll(self):
        def objective(theta):
          val = -self.log_marginal_likelihood(theta)
          # print(f"Trying length_scale={np.exp(theta[0]).item():.4f}, noise={np.exp(theta[1]).item():.6f}, rbf_variance={np.exp(theta[2]).item():.4f} => Obj={val.item():.4f}")
          return val

        initial_theta = np.log([self.length_scale, self.noise, self.rbf_variance])
        res = minimize(objective, initial_theta, method='L-BFGS-B',
                       bounds=[(np.log(1e-2), np.log(1e2)),     # length_scale
                               (np.log(1e-5), np.log(1.0)),     # noise
                               (np.log(1), np.log(1e2))])    # rbf_variance

        self.length_scale, self.noise, self.rbf_variance = np.exp(res.x)
        print(f"Optimized length_scale: {self.length_scale:.4f}, noise: {self.noise:.6f}, rbf_variance: {self.rbf_variance:.4f}")
        self.fit(self.X_train, self.y_train)

    def loo_cv(self, length_scale, rbf_variance, noise):
        """Efficient Leave-One-Out cross-validation predictions"""
        loo_K = self.rbf_kernel(self.X_train, self.X_train, length_scale, rbf_variance)
        loo_K_noise = loo_K + noise * np.eye(len(self.X_train)) + 1e-6 * np.eye(len(self.X_train))
        loo_K_noise_inv = np.linalg.inv(loo_K_noise)
        loo_K_noise_inv_diag = np.diag(loo_K_noise_inv).reshape(-1,1)

        # Compute LOO predictions
        loo_mean = self.y_train - loo_K_noise_inv @ self.y_train / loo_K_noise_inv_diag
        loo_var = 1 / loo_K_noise_inv_diag

        # print('loo_var', loo_var.shape)
        # print('loo_mean', loo_mean.shape)
        # print('self.y_train', self.y_train.shape)

        # predictive_log_prob = 0
        # for i in range(len(loo_mean)):
        #     loo_mean_i = self.y_train[i] - (loo_K_noise_inv @ self.y_train)[i] / loo_K_noise_inv_diag[i]
        #     loo_var_i = 1 / loo_K_noise_inv_diag[i]
        #     ind_plp = - 0.5 * (loo_mean_i - self.y_train[i])**2 / loo_var_i - 0.5 * np.log(loo_var_i) - 0.5 * np.log(np.pi*2)
        #     predictive_log_prob += ind_plp

        predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_train)**2/loo_var - 0.5 * np.log(np.pi * 2)

        return np.sum(predictive_log_prob)
    
    def optimize_loo_cv(self):
        def objective(theta):
            length_scale, noise, rbf_variance = np.exp(theta)
            val = -self.loo_cv(length_scale, rbf_variance, noise)
            print(val)
            return val

        initial_theta = np.log([self.length_scale, self.noise, self.rbf_variance])
        res = minimize(objective, initial_theta, method='L-BFGS-B',
                    #    bounds=[(np.log(1e-2), np.log(1e2)),     # length_scale
                    #            (np.log(1e-3), np.log(1.0)),     # noise
                    #            (np.log(1e-1), np.log(1e2))]    # rbf_variance
        )

        self.length_scale, self.noise, self.rbf_variance = np.exp(res.x)
        print(f"Optimized length_scale: {self.length_scale:.4f}, noise: {self.noise:.6f}, rbf_variance: {self.rbf_variance:.4f}")
        self.fit(self.X_train, self.y_train)

class RCGPRegressor:
    def __init__(self, mean, length_scale=1.0, rbf_variance=1.0, noise=1e-2, epsilon = 0.05):
        self.mean = mean
        self.length_scale = length_scale
        self.rbf_variance = rbf_variance
        self.noise = noise
        self.epsilon = epsilon

    def rbf_kernel(self, X1, X2, length_scale, variance):
        """Compute the RBF kernel with variance (amplitude squared)"""
        dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        return variance * np.exp(-0.5 * dists / length_scale**2)

    # def imq_kernel(self, y, x, beta, c):
    #     return beta * (1 + ((y-x)**2)/(c**2))**-0.5

    # def imq_gradient_log(self, y, x, beta, c):
    #     return 2 * (x - y)/(c**2) * (1+(y-x)**2/(c**2))**-1

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.mean_train = self.mean(X_train)

        self.beta = (self.noise / 2)**0.5
        self.c = np.nanquantile(np.abs(y_train - self.mean_train), 1 - self.epsilon)

        self.w, self.imq_gradient_log_squared = imq_kernel(y_train, self.mean_train, self.beta, self.c)
        # print(self.w.reshape(-1)/beta)

        self.mw = self.mean_train + self.noise * self.imq_gradient_log_squared
        self.Jw = (self.noise/2) * np.diag((self.w**-2).flatten())

        self.K = self.rbf_kernel(X_train, X_train, self.length_scale, self.rbf_variance)
        self.Kw = self.K + self.noise * self.Jw + 1e-6 * np.eye(len(X_train))

        y_centered = y_train - self.mw
        self.L = cholesky(self.Kw)
        self.alpha = solve(self.L.T, solve(self.L, y_centered))

    def predict(self, X_test):
        K_s = self.rbf_kernel(self.X_train, X_test, self.length_scale, self.rbf_variance)
        K_ss = self.rbf_kernel(X_test, X_test, self.length_scale, self.rbf_variance) + 1e-6 * np.eye(len(X_test))

        # mu = self.mean + K_s.T @ self.alpha
        # v = solve(self.L, K_s)
        # cov = K_ss - v.T @ v

        mu = self.mean(X_test) + K_s.T @ np.linalg.inv(self.Kw) @ (self.y_train - self.mw)
        cov = K_ss - K_s.T @ np.linalg.inv(self.Kw) @ K_s
        return mu.reshape(-1), np.diag(cov)
    
    def loo_cv(self, length_scale, rbf_variance, noise, weighted=False):
        """Efficient Leave-One-Out cross-validation predictions"""
        loo_K = self.rbf_kernel(self.X_train, self.X_train, length_scale, rbf_variance)

        beta = (noise / 2)**0.5
        loo_w, loo_gradient_log_squared = imq_kernel(self.y_train, self.mean_train, beta, self.c)

        loo_Jw = (noise/2) * np.diag((loo_w**-2).flatten())
        loo_Kw = loo_K + noise * loo_Jw + 1e-6 * np.eye(len(self.X_train))
        loo_Kw_inv = np.linalg.inv(loo_Kw)
        loo_Kw_inv_diag = np.diag(loo_Kw_inv).reshape(-1,1)

        z = self.y_train - self.mean_train - noise * loo_gradient_log_squared

        # Compute LOO predictions
        loo_mean = z + self.mean_train - loo_Kw_inv @ z / loo_Kw_inv_diag
        loo_var = (1 / loo_Kw_inv_diag) - (noise**2 / 2) * loo_w**-2 + noise

        self.predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_train)**2/loo_var - 0.5 * np.log(np.pi * 2)

        if weighted:
            self.weight = (1 + (self.y_train - self.mean_train)**2/(self.c**2))**-0.5
            result = np.dot(self.predictive_log_prob.flatten(), (self.weight).flatten())
        else:
            result = np.sum(self.predictive_log_prob)
        return result
    
    def optimize_loo_cv(self, weighted=False, print_opt_param = False, print_iter_param=False):
        def objective(theta):
            length_scale, noise, rbf_variance = np.exp(theta)
            val = -self.loo_cv(length_scale, rbf_variance, noise, weighted=weighted)
            if print_iter_param:
                print(-val)
            return val

        initial_theta = np.log([self.length_scale, self.noise, self.rbf_variance])
        res = minimize(objective, initial_theta, method='L-BFGS-B',
                    #    bounds=[(np.log(1e-6), np.log(1e6)),     # length_scale
                    #            (np.log(1e-6), np.log(1e6)),     # noise
                    #            (np.log(1e-6), np.log(1e6))]    # rbf_variance
        )

        self.length_scale, self.noise, self.rbf_variance = np.exp(res.x)
        if print_opt_param:
            print(f"Optimized length_scale: {self.length_scale:.4f}, noise: {self.noise:.6f}, rbf_variance: {self.rbf_variance:.4f}")
        self.fit(self.X_train, self.y_train)