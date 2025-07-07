import numpy as np
from rcgp.kernels import RBFKernel
from scipy.optimize import minimize
from numpy.linalg import cholesky, solve

def imq_kernel(y, x, beta, c):
    # Element-wise difference squared
    y, x, beta, c = y.reshape(-1), x.reshape(-1), beta.reshape(-1), c.reshape(-1)
    diff_sq = (y - x)**2
    denom = (1 + diff_sq / (c**2))
    # Compute the kernel and gradient where both x and y are not NaN
    valid = ~np.isnan(x) & ~np.isnan(y)

    # Initialize outputs with NaN
    imq = np.full_like(x, np.nan, dtype=np.float64)
    gradient_log_squared = np.full_like(x, np.nan, dtype=np.float64)

    # Compute only where valid
    imq[valid] = beta * denom[valid]**(-0.5)
    gradient_log_squared[valid] = 2 * (x[valid] - y[valid]) / (c[valid]**2) * denom[valid]**(-1)

    return imq.reshape(-1,1), gradient_log_squared.reshape(-1,1)

def extract_and_remove_dth(matrix, d):
    row_without_diag = np.delete(matrix[d, :], d)
    diag_elem = matrix[d, d]
    reduced_matrix = np.delete(np.delete(matrix, d, axis=0), d, axis=1)

    return row_without_diag, diag_elem, reduced_matrix

def cross_channel_predictive(Y_train, mean, B, noise_matrix):
        N, D = Y_train.shape
        B_noise = B + noise_matrix
        predictive_means, predictive_variances = np.zeros(Y_train.shape), np.zeros(Y_train.shape)
        
        for i in range(N):
            row = Y_train[i, :]
            for d in range(D):
                if np.isnan(row[d]):
                    predictive_means[i, d] = np.nan
                    predictive_variances[i, d] = np.nan
                else:
                    obs_other = np.delete(row, d)
                    B_d_other, B_dd, B_other_other = extract_and_remove_dth(B_noise, d)

                    # Mask to filter out NaNs
                    mask = ~np.isnan(obs_other)
                    if not np.any(mask):
                        # If all values in obs_other are NaN
                        conditional_mean = mean
                        conditional_variance = B_dd
                    else:
                        B_d_other_masked = B_d_other[mask]
                        B_other_other_masked = B_other_other[np.ix_(mask, mask)]
                        obs_other_masked = obs_other[mask]

                        conditional_mean = (
                            mean +
                            B_d_other_masked.reshape(1, -1) @
                            np.linalg.inv(B_other_other_masked) @
                            (obs_other_masked.reshape(-1, 1) - mean)
                        ).item()
                        conditional_variance = (
                            B_dd -
                            B_d_other_masked.reshape(1, -1) @
                            np.linalg.inv(B_other_other_masked) @
                            B_d_other_masked.reshape(-1, 1)
                        ).item()

                    predictive_means[i, d] = conditional_mean
                    predictive_variances[i, d] = conditional_variance

        return predictive_means, predictive_variances


class MOGPRegressor:
    def __init__(self, mean=0.0, length_scale=1.0, noise=np.array([1e-2]), A=None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise = noise
        self.noise_constraint = (noise.shape[0] == 1)
        if self.noise_constraint:
            self.noise_matrix = noise * np.eye(self.D)
        else:
            if isinstance(noise, np.ndarray) and noise.ndim == 1 and noise.shape[0] == self.D:
                self.noise_matrix = np.diag(noise)
            else:
                raise ValueError(f"`noise` must be a 1D NumPy array of length {self.D} when noise_constraint is False.")
        self.A = A
        self.B = A @ A.T

    def rbf_kernel(self, X1, X2, length_scale):
        dists = np.sum(X1**2, axis=1)[:, None] + \
                np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        return np.exp(-0.5 * dists / length_scale**2)

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N = len(X_train)

        # Flatten Y_train.T and mask out NaNs
        y_vec = Y_train.T.flatten()
        self.mask = ~np.isnan(y_vec)
        self.valid_idx = np.where(self.mask)[0]
        y_vec = y_vec[self.mask].reshape(-1, 1)
        
        self.y_vec = y_vec
        # Kernel matrix for all outputs
        full_K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        noise_K = np.kron(self.noise_matrix, np.eye(self.N))
        K = full_K + noise_K + 1e-6 * np.eye(self.D * self.N)

        # Subset kernel matrix and solve only for valid indices
        self.K_noise = K[np.ix_(self.mask, self.mask)]
        y_centered = self.y_vec - self.mean

        self.L = cholesky(self.K_noise)
        self.alpha = solve(self.L.T, solve(self.L, y_centered))

    def predict(self, X_test):
        N_test = len(X_test)
        K_s = np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale))
        K_s = K_s[self.valid_idx, :]  # Subset only rows corresponding to observed outputs

        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + \
               1e-6 * np.eye(N_test * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v
        var = np.maximum(0, np.diag(cov))  # Avoid small negative variances due to numerical errors

        mu = mu.reshape(self.D, -1).T
        var = var.reshape(self.D, -1).T

        return mu, var

    def log_marginal_likelihood(self, theta=None):
        if theta is not None:
            length_scale = np.exp(theta)[0]
            if self.noise_constraint:
                noise = np.exp(theta)[1]
                noise_matrix = noise * np.eye(self.D)
                A = theta[2:].reshape(self.D, -1)
            else:
                noise = np.exp(theta)[1:self.D+1]
                noise_matrix = np.diag(noise)
                A = theta[self.D+1:].reshape(self.D, -1)
            B = A @ A.T
            full_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))
            noise_K = np.kron(noise_matrix, np.eye(self.N))
            K = full_K + noise_K + 1e-6 * np.eye(self.D * self.N)
        else:
            K = self.K_noise
        K = K[np.ix_(self.valid_idx, self.valid_idx)]
        y_centered = self.y_vec - self.mean

        try:
            L = cholesky(K)
        except np.linalg.LinAlgError:
            return 1e6

        alpha = solve(L.T, solve(L, y_centered))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        result = -0.5 * y_centered.T @ alpha - 0.5 * log_det - 0.5 * len(y_centered) * np.log(2 * np.pi)
        return result.item()

    def optimize_hyperparameters(self):
        def objective(theta):
            return -self.log_marginal_likelihood(theta)

        initial_theta = np.concatenate((
            np.log([self.length_scale]),
            np.log(self.noise).ravel(),
            self.A.reshape(-1)
        ))

        # bounds = [
        #     (np.log(1e-2), np.log(1e2)),     # length_scale
        #     (np.log(1e-5), np.log(1.0)),     # noise
        # ] + [(np.log(1e-1), np.log(5))] * len(self.a)

        res = minimize(objective, initial_theta, method='L-BFGS-B')

        self.length_scale = np.exp(res.x[0])
        if self.noise_constraint:
            self.noise = np.exp(res.x)[1]
            self.noise_matrix = self.noise * np.eye(self.D)
            self.A = res.x[2:].reshape(self.D, -1)
        else:
            self.noise = np.exp(res.x)[1:self.D+1]
            self.noise_matrix = np.diag(self.noise)
            self.A = res.x[self.D+1:].reshape(self.D, -1)
        self.B = self.A @ self.A.T

        print(f"Optimized length_scale: {self.length_scale:.4f}")
        print(f"Optimized noise: {self.noise}")
        print(f"Optimized A: {self.A}")
        print(f"Optimized B: \n{self.B}")

        self.fit(self.X_train, self.Y_train)

    def loo_cv(self, length_scale, noise_matrix, A):
        B = A @ A.T
        loo_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))
        loo_K_noise = loo_K + np.kron(noise_matrix, np.eye(self.N)) + 1e-6 * np.eye(self.D * self.N)
        loo_K_noise_inv = np.linalg.inv(loo_K_noise[np.ix_(self.mask, self.mask)])
        loo_K_noise_inv_diag = np.diag(loo_K_noise_inv).reshape(-1,1)

        # Compute LOO predictions
        loo_mean = self.y_vec - loo_K_noise_inv @ self.y_vec / loo_K_noise_inv_diag
        loo_var = 1 / loo_K_noise_inv_diag

        # print('loo_var', loo_var.shape)
        # print('loo_mean', loo_mean.shape)
        # print('self.y_train', self.y_train.shape)

        predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        return np.sum(predictive_log_prob)
    
    def optimize_loo_cv(self, print_opt_param=False, print_iter_param=False):
        def objective(theta):
            length_scale = np.exp(theta)[0]
            if self.noise_constraint:
                noise = np.exp(theta)[1]
                noise_matrix = noise * np.eye(self.D)
                A = theta[2:].reshape(self.D, -1)
            else:
                noise = np.exp(theta)[1:self.D+1]
                noise_matrix = np.diag(noise)
                A = theta[self.D+1:].reshape(self.D, -1)
            val = -self.loo_cv(length_scale, noise_matrix, A)
            if print_iter_param:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale]),
            np.log(self.noise),
            self.A.reshape(-1)
        ))
        res = minimize(objective, initial_theta, method='L-BFGS-B',
                    #    bounds=[(np.log(1e-2), np.log(1e2)),     # length_scale
                    #            (np.log(1e-3), np.log(1.0)),     # noise
                    #            (np.log(1e-1), np.log(1e2))]    # rbf_variance
        )

        self.length_scale = np.exp(res.x[0])
        if self.noise_constraint:
            self.noise = np.exp(res.x)[1]
            self.noise_matrix = self.noise * np.eye(self.D)
            self.A = res.x[2:].reshape(self.D, -1)
        else:
            self.noise = np.exp(res.x)[1:self.D+1]
            self.noise_matrix = np.diag(self.noise)
            self.A = res.x[self.D+1:].reshape(self.D, -1)
        self.B = self.A @ self.A.T

        if print_opt_param:
            print(f"Optimized length_scale: {self.length_scale:.4f}")
            print(f"Optimized noise: {self.noise}")
            print(f"Optimized A: {self.A}")
            print(f"Optimized B: \n{self.B}")

        self.fit(self.X_train, self.Y_train)


class MORCGPRegressor_PM:
    def __init__(self, n_outputs, mean=0.0, length_scale=1.0, noise=1e-2, A = None, epsilon = 0.05):
        self.D = n_outputs
        self.mean = mean
        self.length_scale = length_scale
        self.noise = noise
        self.A = A
        self.B = A @ A.T
        self.epsilon = epsilon

    def rbf_kernel(self, X1, X2, length_scale):
        """Compute the RBF kernel with variance (amplitude squared)"""
        dists = np.sum(X1**2, axis=1)[:, None] + \
                np.sum(X2**2, axis=1)[None, :] - \
                2 * X1 @ X2.T
        return np.exp(-0.5 * dists / length_scale**2)

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N, _ = Y_train.shape

        y_vec = Y_train.T.flatten()
        mask = ~np.isnan(y_vec)
        self.valid_idx = np.where(mask)[0]
        self.y_vec = y_vec.reshape(-1, 1)[mask,:]

        beta = (self.noise / 2)**0.5
        c = np.nanquantile(self.Y_train, 1 - self.epsilon, axis=0).reshape(-1,1)
        c_full = np.kron(c, np.ones((self.N, 1))) # shape (200, 1)
        c_valid = c_full[mask]

        w_valid, gradient_log_squared_valid = imq_kernel(self.y_vec, self.mean, beta, c_valid)
        w, gradient_log_squared = np.full((self.N*self.D, 1), np.nan), np.full((self.N*self.D, 1), np.nan)
        w[mask] = w_valid
        gradient_log_squared[mask] = gradient_log_squared_valid

        self.mw = self.mean + self.noise * gradient_log_squared
        self.Jw = (self.noise/2) * np.diag((w**-2).flatten())

        self.K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        self.Kw = (self.K + np.kron(self.noise * np.eye(self.D), np.eye(self.N)) @ self.Jw + 1e-6 * np.eye(self.D * self.N))[np.ix_(mask, mask)]

        y_centered_w = self.y_vec - self.mw[mask, :]

        self.L = cholesky(self.Kw)
        self.alpha = solve(self.L.T, solve(self.L, y_centered_w))

    def predict(self, X_test):
        K_s = (np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale)))[self.valid_idx, :]
        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + 1e-6 * np.eye(len(X_test) * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v
        std = np.sqrt(np.diag(cov))

        mu = mu.reshape(self.D, -1).T
        std = np.sqrt(np.diag(cov)).reshape(self.D, -1).T

        return mu, std

class MORCGPRegressor:
    def __init__(self, mean=0.0, length_scale=1.0, noise=np.array([1e-2]), A = None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise = noise
        self.noise_constraint = (noise.shape[0] == 1)
        if self.noise_constraint:
            self.noise_matrix = noise * np.eye(self.D)
        else:
            if isinstance(noise, np.ndarray) and noise.ndim == 1 and noise.shape[0] == self.D:
                self.noise_matrix = np.diag(noise)
            else:
                raise ValueError(f"`noise` must be a 1D NumPy array of length {self.D} when noise_constraint is False.")
        self.A = A
        self.B = A @ A.T

    def rbf_kernel(self, X1, X2, length_scale):
        """Compute the RBF kernel with variance (amplitude squared)"""
        dists = np.sum(X1**2, axis=1)[:, None] + \
                np.sum(X2**2, axis=1)[None, :] - \
                2 * X1 @ X2.T
        return np.exp(-0.5 * dists / length_scale**2)

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N, _ = Y_train.shape

        y_vec = Y_train.T.flatten()
        self.mask = ~np.isnan(y_vec)
        self.valid_idx = np.where(self.mask)[0]
        self.y_vec = y_vec.reshape(-1, 1)[self.mask,:]

        if self.noise_constraint:
            noise_vec = self.noise * np.ones(self.N * self.D)
        else:
            noise_vec = np.kron(self.noise, np.ones(self.N))
        beta = (noise_vec / 2)**0.5

        predictive_means, predictive_variances = cross_channel_predictive(Y_train, self.mean, self.B, self.noise_matrix)
        self.w, gradient_log_squared = imq_kernel(y_vec, predictive_means.reshape((-1,1), order='F'), beta, np.sqrt(predictive_variances).reshape((-1,1), order='F'))

        self.mw = self.mean + noise_vec.reshape(-1,1) * gradient_log_squared
        self.Jw = np.diag((noise_vec/2)) @ np.diag((self.w**-2).flatten())

        self.K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        self.Kw = (self.K + np.kron(self.noise_matrix, np.eye(self.N)) @ self.Jw + 1e-6 * np.eye(self.D * self.N))[np.ix_(self.mask, self.mask)]

        y_centered_w = self.y_vec - self.mw[self.mask, :]

        self.L = cholesky(self.Kw)
        self.alpha = solve(self.L.T, solve(self.L, y_centered_w))

        return predictive_means, predictive_variances

    def predict(self, X_test):
        K_s = (np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale)))[self.valid_idx, :]
        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + 1e-6 * np.eye(len(X_test) * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v
        std = np.sqrt(np.diag(cov))

        # print('mu.shape', mu.shape)
        # print('self.alpha.shape', self.alpha.shape)

        mu = mu.reshape(self.D, -1).T
        std = np.sqrt(np.diag(cov)).reshape(self.D, -1).T

        return mu, std
    
    def loo_cv(self, length_scale, noise_matrix, A, weighted=False, B_weighted=None):
        
        B = A @ A.T
        loo_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))

        predictive_means, predictive_variances = cross_channel_predictive(self.Y_train, self.mean, B, noise_matrix)

        beta = np.kron((np.diag(noise_matrix) / 2)**0.5, np.ones(self.N))

        loo_w, loo_gradient_log_squared = imq_kernel(self.Y_train.T.flatten(), predictive_means.reshape((-1,1), order='F'), beta, (np.sqrt(predictive_variances)).reshape((-1,1), order='F'))
        loo_Jw = (noise/2) * np.diag((loo_w**-2).flatten())
        loo_Kw = loo_K + noise * loo_Jw + 1e-6 * np.eye(self.D * self.N)
        loo_Kw_inv = np.linalg.inv(loo_Kw[np.ix_(self.mask, self.mask)])
        loo_Kw_inv_diag = np.diag(loo_Kw_inv).reshape(-1,1)

        z = self.y_vec - self.mean - noise * loo_gradient_log_squared[self.valid_idx,:]

        # Compute LOO predictions
        loo_mean = z + self.mean - loo_Kw_inv @ z / loo_Kw_inv_diag
        # print('loo_Kw_inv_diag.shape', loo_Kw_inv_diag.shape)
        # print('self.w.shape', self.w.shape)
        loo_var = (1 / loo_Kw_inv_diag) - (noise**4 / 2) * (self.w**-2)[self.valid_idx,:] + noise**2

        self.predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        if weighted:
            pred_means_loo, pred_var_loo = cross_channel_predictive(Y_train=self.Y_train, mean=self.mean, B=B_weighted, noise=noise)
            weights, _ = imq_kernel(self.Y_train.T.flatten(), pred_means_loo.reshape((-1,1), order='F'), beta, np.sqrt(pred_var_loo).reshape((-1,1), order='F'))
            self.weights_01 = weights[self.valid_idx,:]/beta
            result = np.dot(self.predictive_log_prob.flatten(), self.weights_01.flatten())
        else:
            result = np.sum(self.predictive_log_prob)
        return result
    
    def optimize_loo_cv(self, weighted=False, print_opt_param = False, print_iter_param=False, B_weighted=None):
        def objective(theta):
            length_scale, noise = np.exp(theta[:2])
            A = theta[2:].reshape(self.D, -1)
            if weighted:
                val = -self.loo_cv(length_scale, noise, A, weighted=True, B_weighted=B_weighted)
            else:
                val = -self.loo_cv(length_scale, noise, A, weighted=False)
            if print_iter_param:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale, self.noise]),
            self.A.reshape(-1)
        ))
        res = minimize(objective, initial_theta, method='L-BFGS-B', options={'ftol': 1e-7},
                    #    bounds=[(np.log(1e-2), np.log(1e2)),     # length_scale
                    #            (np.log(1e-3), np.log(1.0)),     # noise
                    #            (np.log(1e-1), np.log(1e2))]    # rbf_variance
        )

        self.length_scale = np.exp(res.x[0])
        self.noise = np.exp(res.x[1])
        self.A = res.x[2:].reshape(self.D,-1)
        self.B = self.A @ self.A.T

        if print_opt_param:
            print(f"Optimized length_scale: {self.length_scale:.4f}, noise: {self.noise:.6f}")
            print(f"Optimized A: {self.A}")
            print(f"Optimized B: \n{self.B}")

        self.fit(self.X_train, self.Y_train)