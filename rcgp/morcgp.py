import numpy as np
from rcgp.kernels import RBFKernel
from scipy.optimize import minimize
from numpy.linalg import cholesky, solve
from scipy.linalg import cho_factor, cho_solve
from numba import jit

def imq_kernel(y, x, beta, c):
    # Element-wise difference squared
    if np.isscalar(beta):
        beta = np.full_like(y, beta)
    if np.isscalar(c):
        c = np.full_like(y, c)
    if np.isscalar(x):
        x = np.full_like(y, x)
    y, x, beta, c = y.reshape(-1), x.reshape(-1), beta.reshape(-1), c.reshape(-1)
    diff_sq = (y - x)**2
    denom = (1 + diff_sq / (c**2))
    # Compute the kernel and gradient where both x and y are not NaN
    valid = ~np.isnan(x) & ~np.isnan(y)

    # Initialize outputs with NaN
    imq = np.full_like(x, np.nan, dtype=np.float64)
    gradient_log_squared = np.full_like(x, np.nan, dtype=np.float64)

    # Compute only where valid
    imq[valid] = beta[valid] * denom[valid]**(-0.5)
    gradient_log_squared[valid] = 2 * (x[valid] - y[valid]) / (c[valid]**2) * denom[valid]**(-1)

    return imq.reshape(-1,1), gradient_log_squared.reshape(-1,1)

def scaled_imq_weight(y, x, c):
    if np.isscalar(c):
        c = np.full_like(y, c)
    if np.isscalar(x):
        x = np.full_like(y, x)

    # Element-wise difference squared
    y, x, c = y.reshape(-1), x.reshape(-1), c.reshape(-1)
    diff_sq = (y - x)**2
    denom = (1 + diff_sq / (c**2))
    # Compute the kernel and gradient where both x and y are not NaN
    valid = ~np.isnan(x) & ~np.isnan(y)

    # Initialize outputs with NaN
    imq = np.full_like(x, np.nan, dtype=np.float64)
    gradient_log_squared = np.full_like(x, np.nan, dtype=np.float64)

    # Compute only where valid
    imq[valid] = denom[valid]**(-0.5)
    gradient_log_squared[valid] = 2 * (x[valid] - y[valid]) / (c[valid]**2) * denom[valid]**(-1)

    return imq.reshape(-1,1), gradient_log_squared.reshape(-1,1)

def extract_and_remove_dth(matrix, d):
    row_without_diag = np.delete(matrix[d, :], d)
    diag_elem = matrix[d, d]
    reduced_matrix = np.delete(np.delete(matrix, d, axis=0), d, axis=1)

    return row_without_diag, diag_elem, reduced_matrix

def cross_channel_predictive(Y_train, mean, B, noise):
        N, D = Y_train.shape
        if np.isscalar(noise):
            noise_matrix = noise * np.eye(D)
        else:
            noise_matrix = np.diag(noise.flatten())
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
                        B_other_other_masked = B_other_other[np.ix_(mask, mask)] + 1e-3 * np.eye(len(B_other_other[np.ix_(mask, mask)]))
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

# @jit(nopython=True, fastmath=True)
def rbf_kernel(X1, X2, length_scale):
    X1 = np.ascontiguousarray(X1)
    X2 = np.ascontiguousarray(X2)
    
    dists = np.sum(X1**2, axis=1)[:, None] + \
            np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    return np.exp(-0.5 * dists / length_scale**2)

class MOGPRegressor:
    def __init__(self, mean=0.0, length_scale=1.0, noise=None, A=None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise = noise
        self.noise_matrix = np.diag(noise)
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
        self.noise_matrix = np.diag(self.noise)

        y_vec = Y_train.T.flatten()
        self.mask = ~np.isnan(y_vec)
        self.valid_idx = np.where(self.mask)[0]
        y_vec = y_vec[self.mask].reshape(-1, 1)
        
        self.y_vec = y_vec
        full_K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        noise_K = np.kron(self.noise_matrix, np.eye(self.N))
        K = full_K + noise_K + 1e-6 * np.eye(self.D * self.N)

        self.K_noise = K[np.ix_(self.mask, self.mask)]
        y_centered = self.y_vec - self.mean

        self.L = cholesky(self.K_noise)
        self.alpha = solve(self.L.T, solve(self.L, y_centered))

    def predict(self, X_test):
        N_test = len(X_test)
        K_s = np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale))
        K_s = K_s[self.valid_idx, :] 

        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + \
               1e-6 * np.eye(N_test * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v
        var = np.maximum(0, np.diag(cov))

        mu = mu.reshape(self.D, -1).T
        var = var.reshape(self.D, -1).T

        return mu, var

    def log_marginal_likelihood(self, theta=None):
        if theta is not None:
            length_scale = np.exp(theta)[0]
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

    def optimize_hyperparameters(self, print_opt_param=False, print_iter_param=False):
        def objective(theta):
            val = -self.log_marginal_likelihood(theta)
            if print_iter_param:
                print(-val)
            return val

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

    def loo_cv(self, length_scale, noise, A):
        B = A @ A.T
        loo_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))
        loo_K_noise = loo_K + np.kron(np.diag(noise), np.eye(self.N)) + 1e-6 * np.eye(self.D * self.N)
        loo_K_noise_inv = np.linalg.inv(loo_K_noise[np.ix_(self.mask, self.mask)])
        loo_K_noise_inv_diag = np.diag(loo_K_noise_inv).reshape(-1,1)

        # Compute LOO predictions
        loo_mean = self.y_vec - loo_K_noise_inv @ self.y_vec / loo_K_noise_inv_diag
        loo_var = 1 / loo_K_noise_inv_diag

        predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        return np.sum(predictive_log_prob)
    
    def optimize_loo_cv(self, print_opt_param=False, print_iter_param=False):
        def objective(theta):
            length_scale = np.exp(theta)[0]
            noise = np.exp(theta)[1:self.D+1]
            A = theta[self.D+1:].reshape(self.D, -1)
            val = -self.loo_cv(length_scale, noise, A)
            if print_iter_param:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale]),
            np.log(self.noise),
            self.A.reshape(-1)
        ))
        res = minimize(objective, initial_theta, method='L-BFGS-B', tol=1e-2,
                    #    bounds=[(np.log(1e-2), np.log(1e2)),     # length_scale
                    #            (np.log(1e-3), np.log(1.0)),     # noise
                    #            (np.log(1e-1), np.log(1e2))]    # rbf_variance
        )

        self.length_scale = np.exp(res.x[0])
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

class MOGPRegressor_NC:
    def __init__(self, mean=0.0, length_scale=1.0, noise=1e-2, A=None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise = noise
        self.A = A
        self.B = A @ A.T

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N = len(X_train)

        y_vec = Y_train.T.flatten()
        self.mask = ~np.isnan(y_vec)
        self.valid_idx = np.where(self.mask)[0]
        y_vec = y_vec[self.mask].reshape(-1, 1)
        
        self.y_vec = y_vec
        full_K = np.kron(self.B, rbf_kernel(X_train, X_train, self.length_scale))
        noise_K = np.kron(self.noise * np.eye(self.D), np.eye(self.N))
        K = full_K + noise_K + 1e-6 * np.eye(self.D * self.N)

        self.K_noise = K[np.ix_(self.mask, self.mask)]
        y_centered = self.y_vec - self.mean

        self.L = cholesky(self.K_noise)
        self.alpha = solve(self.L.T, solve(self.L, y_centered))


    def predict(self, X_test):
        N_test = len(X_test)
        K_s = np.kron(self.B, rbf_kernel(self.X_train, X_test, self.length_scale))
        K_s = K_s[self.valid_idx, :]

        K_ss = np.kron(self.B, rbf_kernel(X_test, X_test, self.length_scale)) + \
               1e-6 * np.eye(N_test * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v
        var = np.maximum(0, np.diag(cov)) 

        mu = mu.reshape(self.D, -1).T
        var = var.reshape(self.D, -1).T

        return mu, var

    def log_marginal_likelihood(self, theta=None):
        if theta is not None:
            length_scale = np.exp(theta)[0]
            noise = np.exp(theta)[1]
            A = theta[2:].reshape(self.D, -1)
            B = A @ A.T
            full_K = np.kron(B, rbf_kernel(self.X_train, self.X_train, length_scale))
            noise_K = np.kron(noise * np.eye(self.D), np.eye(self.N))
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

    def optimize_hyperparameters(self, print_opt_param=False, print_iter_param=False):
        def objective(theta):
            val = -self.log_marginal_likelihood(theta)
            if print_iter_param:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale, self.noise]),
            self.A.reshape(-1)
        ))

        res = minimize(objective, initial_theta, method='L-BFGS-B', tol=1e-2)

        self.length_scale = np.exp(res.x[0])
        self.noise = np.exp(res.x[1])
        self.A = res.x[2:].reshape(self.D,-1)
        self.B = self.A @ self.A.T

        if print_opt_param:
            print(f"Optimized length_scale: {self.length_scale:.4f}, noise: {self.noise:.6f}")
            print(f"Optimized A: {self.A}")
            print(f"Optimized B: \n{self.B}")

        self.fit(self.X_train, self.Y_train)

    def loo_cv(self, length_scale, noise, A):
        B = A @ A.T
        loo_K = np.kron(B, rbf_kernel(self.X_train, self.X_train, length_scale))
        loo_K_noise = loo_K + noise * np.eye(self.D * self.N) + 1e-6 * np.eye(self.D * self.N)
        loo_K_noise_inv = np.linalg.inv(loo_K_noise[np.ix_(self.mask, self.mask)])
        loo_K_noise_inv_diag = np.diag(loo_K_noise_inv).reshape(-1,1)

        # Compute LOO predictions
        loo_mean = self.y_vec - loo_K_noise_inv @ self.y_vec / loo_K_noise_inv_diag
        loo_var = 1 / loo_K_noise_inv_diag

        predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        return np.sum(predictive_log_prob)

class MORCGPRegressor_PM:
    def __init__(self, mean=0.0, length_scale=1.0, noise=np.array([1e-2]), A = None, epsilons=None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise = noise
        self.noise_matrix = np.diag(noise)
        self.A = A
        self.B = A @ A.T
        self.epsilons = epsilons

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

        noise_vec = np.kron((self.noise).reshape(-1,1), np.ones((self.N, 1)))
        beta = (noise_vec / 2)**0.5

        c = np.array([
            np.nanquantile(self.Y_train[:, d], self.epsilons[d])
            for d in range(self.D)
        ])
        self.cs = np.repeat(c, self.N).reshape(-1, 1)
        self.w, gradient_log_squared = imq_kernel(y_vec, self.mean, beta, self.cs)
        self.w01_plot = self.w/beta
        self.w01 = self.w[self.valid_idx,:] / beta[self.valid_idx,:]

        self.mw = self.mean + noise_vec * gradient_log_squared
        self.Jw = np.diag((noise_vec.flatten()/2)) @ np.diag((self.w**-2).flatten())
        self.K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        self.Kw = (self.K + np.kron(self.noise_matrix, np.eye(self.N)) @ self.Jw + 1e-6 * np.eye(self.D * self.N))[np.ix_(self.mask, self.mask)]

        y_centered_w = self.y_vec - self.mw[self.mask, :]

        self.L = cholesky(self.Kw)
        self.alpha = solve(self.L.T, solve(self.L, y_centered_w))

    def predict(self, X_test):
        K_s = (np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale)))[self.valid_idx, :]
        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + 1e-6 * np.eye(len(X_test) * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v

        mu = mu.reshape(self.D, -1).T
        var = np.diag(cov).reshape(self.D, -1).T

        return mu, var
    
    def loo_cv(self, length_scale, noise, A, weighted=False):        
        B = A @ A.T
        loo_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))

        noise_matrix = np.diag(noise)

        noise_vec = np.kron(noise.reshape(-1,1), np.ones((self.N,1)))
        beta = (noise_vec / 2)**0.5

        loo_w, loo_gradient_log_squared = imq_kernel(self.Y_train.T.reshape(-1,1), self.mean, beta, self.cs)
        loo_Jw = np.diag((noise_vec.flatten()/2)) @ np.diag((loo_w**-2).flatten())
        loo_Kw = loo_K + np.kron(noise_matrix, np.eye(self.N)) @ loo_Jw + 1e-6 * np.eye(self.D * self.N)
        loo_Kw_inv = np.linalg.inv(loo_Kw[np.ix_(self.mask, self.mask)])
        loo_Kw_inv_diag = np.diag(loo_Kw_inv).reshape(-1,1)
        z = self.y_vec - self.mean - (noise_vec * loo_gradient_log_squared)[self.valid_idx,:]

        # Compute LOO predictions
        loo_mean = z + self.mean - loo_Kw_inv @ z / loo_Kw_inv_diag
        loo_var = (1 / loo_Kw_inv_diag) - ((noise_vec**2 / 2) * (loo_w**-2))[self.valid_idx] + (noise_vec.reshape(-1,1))[self.valid_idx,:]
        self.predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        if weighted:
            result = np.dot(self.predictive_log_prob.flatten(), (self.w01.flatten())**2)
        else:
            result = np.sum(self.predictive_log_prob)
        return result
    
    def optimize_loo_cv(self, weighted=False, print_opt_param = False, print_iter_param=False):
        def objective(theta):
            length_scale = np.exp(theta)[0]
            noise = np.exp(theta)[1:self.D+1]
            A = theta[self.D+1:].reshape(self.D, -1)
            if weighted:
                val = -self.loo_cv(length_scale, noise, A, weighted=True)
            else:
                val = -self.loo_cv(length_scale, noise, A, weighted=False)
            if print_iter_param:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale]),
            np.log(self.noise),
            self.A.reshape(-1)
        ))
        res = minimize(objective, initial_theta, method='L-BFGS-B', tol=1e-2,
                    #    bounds=[(np.log(1e-2), np.log(1e2)),     # length_scale
                    #            (np.log(1e-3), np.log(1.0)),     # noise
                    #            (np.log(1e-1), np.log(1e2))]    # rbf_variance
        )

        self.length_scale = np.exp(res.x[0])
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
    
class MORCGP:
    def __init__(self, mean=0.0, length_scale=1.0, noise_var=np.array([0.1]), A = None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.A = A
        self.B = A @ A.T

    def rbf_kernel(self, X1, X2, length_scale):
        """Compute the RBF kernel with variance (amplitude squared)"""
        dists = np.sum(X1**2, axis=1)[:, None] + \
                np.sum(X2**2, axis=1)[None, :] - \
                2 * X1 @ X2.T
        return np.exp(-0.5 * dists / length_scale**2)

    def fit(self, X_train, Y_train, epsilons=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N, _ = Y_train.shape
        self.epsilons = epsilons

        y_vec = Y_train.T.flatten()
        self.valid_idx = np.where(~np.isnan(y_vec))[0]

        noise_var_vec = np.kron((self.noise_var).reshape(-1,1), np.ones((self.N, 1)))
        predictive_means, predictive_variances = cross_channel_predictive(Y_train, self.mean, self.B, self.noise_var)

        gamma = predictive_means
        # c = 2 * np.sqrt(predictive_variances)
        c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))

        self.w01, gradient_log_squared = scaled_imq_weight(y_vec, gamma.reshape((-1,1), order='F'), c.reshape((-1,1), order='F'))

        self.mw = self.mean + noise_var_vec * gradient_log_squared
        self.Jw = np.diag(self.w01.flatten()**-2)
        self.K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        self.Kw = (self.K + np.kron(np.diag(self.noise_var), np.eye(self.N)) @ self.Jw + 1e-6 * np.eye(self.D * self.N))[np.ix_(self.valid_idx, self.valid_idx)]

        self.y_vec = y_vec.reshape(-1, 1)[self.valid_idx,:]
        y_centered_w = self.y_vec - self.mw[self.valid_idx, :]

        self.L = cholesky(self.Kw)
        self.alpha = solve(self.L.T, solve(self.L, y_centered_w))

        return gamma, c

    def predict(self, X_test):
        K_s = (np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale)))[self.valid_idx, :]
        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + 1e-6 * np.eye(len(X_test) * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v

        mu = mu.reshape(self.D, -1).T
        var = np.diag(cov).reshape(self.D, -1).T

        return mu, var
    
    def loo_cv(self, length_scale, noise_var, A, k=1, fix_weights=True):        
        B = A @ A.T
        loo_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))

        noise_var_vec = np.kron(noise_var.reshape(-1,1), np.ones((self.N,1)))

        if fix_weights:
            loo_w01, loo_gradient_log_squared = self.init_w01, self.init_grad_log2
        else:
            loo_predictive_means, loo_predictive_variances = cross_channel_predictive(self.Y_train, self.mean, B, noise_var)
            loo_gamma = loo_predictive_means
            loo_c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - loo_gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))

            loo_w01, loo_gradient_log_squared = scaled_imq_weight(self.Y_train.T.reshape(-1,1), loo_gamma.reshape((-1,1), order='F'), loo_c)

        loo_Jw = np.diag((loo_w01**-2).flatten())
        loo_Kw = loo_K + np.kron(np.diag(noise_var), np.eye(self.N)) @ loo_Jw + 1e-6 * np.eye(self.D * self.N)
        loo_Kw_inv = np.linalg.inv(loo_Kw[np.ix_(self.valid_idx, self.valid_idx)])
        loo_Kw_inv_diag = np.diag(loo_Kw_inv).reshape(-1,1)
        z = self.y_vec - self.mean - (noise_var_vec * loo_gradient_log_squared)[self.valid_idx,:]

        # Compute LOO predictions
        loo_mean = z + self.mean - loo_Kw_inv @ z / loo_Kw_inv_diag
        loo_var = (1 / loo_Kw_inv_diag) - (noise_var_vec * (loo_w01**-2))[self.valid_idx] + (noise_var_vec.reshape(-1,1))[self.valid_idx,:]
        self.predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        return np.dot(self.predictive_log_prob.flatten(), self.init_w01[self.valid_idx, :].flatten()**k)
    
    def optimize_loo_cv(self, print_opt_param = False, print_iter_objective=False, k=1, init_cov=None, fix_weights=True):

        init_predictive_means, init_predictive_variance = cross_channel_predictive(self.Y_train, self.mean, init_cov, 0)
        init_gamma = init_predictive_means
        init_c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - init_gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))
        self.init_w01, self.init_grad_log2 = scaled_imq_weight(self.Y_train.T.reshape(-1,1), init_gamma.reshape((-1,1), order='F'), init_c.reshape((-1,1), order='F'))

        def objective(theta):
            length_scale = np.exp(theta)[0]
            noise_var = np.exp(theta)[1:self.D+1]
            A = theta[self.D+1:].reshape(self.D, -1)
            val = -self.loo_cv(length_scale, noise_var, A, k=k, fix_weights=fix_weights)
            if print_iter_objective:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale]),
            np.log(self.noise_var),
            self.A.reshape(-1)
        ))
        res = minimize(objective, initial_theta, method='L-BFGS-B', 
                    #    tol=1e-2,
                    #    bounds=[(np.log(1e-2), np.log(1e2)),     # length_scale
                    #            (np.log(1e-3), np.log(1.0)),     # noise_var
                    #            (np.log(1e-1), np.log(1e2))]    # rbf_variance
        )

        self.length_scale = np.exp(res.x[0])
        self.noise_var = np.exp(res.x)[1:self.D+1]
        self.A = res.x[self.D+1:].reshape(self.D, -1)
        self.B = self.A @ self.A.T

        if print_opt_param:
            print(f"Optimized length_scale: {self.length_scale:.4f}")
            print(f"Optimized noise_var: {self.noise_var}")
            print(f"Optimized A: {self.A}")
            print(f"Optimized B: \n{self.B}")

        gamma, c = self.fit(self.X_train, self.Y_train, self.epsilons)
        return init_gamma, init_c, gamma, c

class MORCGP_shared_noise:
    def __init__(self, mean=0.0, length_scale=1.0, noise_var=0.1, A = None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.A = A
        self.B = A @ A.T

    def rbf_kernel(self, X1, X2, length_scale):
        """Compute the RBF kernel with variance (amplitude squared)"""
        dists = np.sum(X1**2, axis=1)[:, None] + \
                np.sum(X2**2, axis=1)[None, :] - \
                2 * X1 @ X2.T
        return np.exp(-0.5 * dists / length_scale**2)

    def fit(self, X_train, Y_train, epsilons=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N, _ = Y_train.shape
        self.epsilons = epsilons

        y_vec = Y_train.T.flatten()
        self.valid_idx = np.where(~np.isnan(y_vec))[0]

        noise_var_vec = self.noise_var * np.ones((self.N*self.D, 1))
        predictive_means, predictive_variances = cross_channel_predictive(Y_train, self.mean, self.B, self.noise_var)

        gamma = predictive_means
        # c = 2 * np.sqrt(predictive_variances)
        c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))

        self.w01, gradient_log_squared = scaled_imq_weight(y_vec, gamma.reshape((-1,1), order='F'), c.reshape((-1,1), order='F'))

        self.mw = self.mean + noise_var_vec * gradient_log_squared
        self.Jw = np.diag(self.w01.flatten()**-2)
        self.K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        self.Kw = (self.K + (self.noise_var * np.eye(self.N * self.D)) @ self.Jw + 1e-6 * np.eye(self.D * self.N))[np.ix_(self.valid_idx, self.valid_idx)]

        self.y_vec = y_vec.reshape(-1, 1)[self.valid_idx,:]
        y_centered_w = self.y_vec - self.mw[self.valid_idx, :]

        self.L = cholesky(self.Kw)
        self.alpha = solve(self.L.T, solve(self.L, y_centered_w))

        return gamma, c

    def predict(self, X_test):
        K_s = (np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale)))[self.valid_idx, :]
        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + 1e-6 * np.eye(len(X_test) * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v

        mu = mu.reshape(self.D, -1).T
        var = np.diag(cov).reshape(self.D, -1).T

        return mu, var
    
    def loo_cv(self, length_scale, noise_var, A, k=1, fix_weights=True):        
        B = A @ A.T
        loo_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))

        noise_var_vec = noise_var * np.ones((self.N * self.D,1))

        if fix_weights:
            loo_w01, loo_gradient_log_squared = self.init_w01, self.init_grad_log2
        else:
            loo_predictive_means, loo_predictive_variances = cross_channel_predictive(self.Y_train, self.mean, B, noise_var)
            loo_gamma = loo_predictive_means
            # loo_c = np.sqrt(loo_predictive_variances).reshape((-1,1), order='F')

            loo_c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - loo_gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))

            loo_w01, loo_gradient_log_squared = scaled_imq_weight(self.Y_train.T.reshape(-1,1), loo_gamma.reshape((-1,1), order='F'), loo_c)

        loo_Jw = np.diag((loo_w01**-2).flatten())
        loo_Kw = loo_K + noise_var * np.eye(self.N*self.D) @ loo_Jw + 1e-4 * np.eye(self.D * self.N)
        loo_Kw_inv = np.linalg.inv(loo_Kw[np.ix_(self.valid_idx, self.valid_idx)])
        loo_Kw_inv_diag = np.diag(loo_Kw_inv).reshape(-1,1)
        z = self.y_vec - self.mean - (noise_var_vec * loo_gradient_log_squared)[self.valid_idx,:]

        # Compute LOO predictions
        loo_mean = z + self.mean - loo_Kw_inv @ z / loo_Kw_inv_diag
        loo_var = (1 / loo_Kw_inv_diag) - (noise_var_vec * (loo_w01**-2))[self.valid_idx] + (noise_var_vec.reshape(-1,1))[self.valid_idx,:]
        self.predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        return np.dot(self.predictive_log_prob.flatten(), self.init_w01[self.valid_idx, :].flatten()**k)
    
    def optimize_loo_cv(self, print_opt_param = False, print_iter_objective=False, k=1, init_cov=None, fix_weights=True):

        init_predictive_means, init_predictive_variance = cross_channel_predictive(self.Y_train, self.mean, init_cov, 0)
        init_gamma = init_predictive_means
        # init_c = np.sqrt(init_predictive_variance).reshape((-1,1), order='F')
        init_c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - init_gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))
        self.init_w01, self.init_grad_log2 = scaled_imq_weight(self.Y_train.T.reshape(-1,1), init_gamma.reshape((-1,1), order='F'), init_c.reshape((-1,1), order='F'))

        def objective(theta):
            length_scale = np.exp(theta)[0]
            noise_var = np.exp(theta)[1]
            A = theta[2:].reshape(self.D, -1)
            val = -self.loo_cv(length_scale, noise_var, A, k=k, fix_weights=fix_weights)
            if print_iter_objective:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale, self.noise_var]),
            self.A.reshape(-1)
        ))
        res = minimize(objective, initial_theta, method='L-BFGS-B', tol=1e-2,
                    #    bounds=[(np.log(1e-2), np.log(1e2)),     # length_scale
                    #            (np.log(1e-3), np.log(1.0)),     # noise_var
                    #            (np.log(1e-1), np.log(1e2))]    # rbf_variance
        )

        self.length_scale = np.exp(res.x[0])
        self.noise_var = np.exp(res.x)[1]
        self.A = res.x[2:].reshape(self.D, -1)
        self.B = self.A @ self.A.T

        if print_opt_param:
            print(f"Optimized length_scale: {self.length_scale:.4f}")
            print(f"Optimized noise_var: {self.noise_var}")
            print(f"Optimized A: {self.A}")
            print(f"Optimized B: \n{self.B}")

        gamma, c = self.fit(self.X_train, self.Y_train, self.epsilons)
        return init_gamma, init_c, gamma, c
    
class MORCGP_bounded_noises:
    def __init__(self, mean=0.0, length_scale=1.0, noise_var=np.array([0.1]), A = None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.A = A
        self.B = A @ A.T

    def rbf_kernel(self, X1, X2, length_scale):
        """Compute the RBF kernel with variance (amplitude squared)"""
        dists = np.sum(X1**2, axis=1)[:, None] + \
                np.sum(X2**2, axis=1)[None, :] - \
                2 * X1 @ X2.T
        return np.exp(-0.5 * dists / length_scale**2)

    def fit(self, X_train, Y_train, epsilons=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N, _ = Y_train.shape
        self.epsilons = epsilons

        y_vec = Y_train.T.flatten()
        self.valid_idx = np.where(~np.isnan(y_vec))[0]

        noise_var_vec = np.kron((self.noise_var).reshape(-1,1), np.ones((self.N, 1)))
        predictive_means, predictive_variances = cross_channel_predictive(Y_train, self.mean, self.B, self.noise_var)

        gamma = predictive_means
        # c = 2 * np.sqrt(predictive_variances)
        c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))

        self.w01, gradient_log_squared = scaled_imq_weight(y_vec, gamma.reshape((-1,1), order='F'), c.reshape((-1,1), order='F'))

        self.mw = self.mean + noise_var_vec * gradient_log_squared
        self.Jw = np.diag(self.w01.flatten()**-2)
        self.K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        self.Kw = (self.K + np.kron(np.diag(self.noise_var), np.eye(self.N)) @ self.Jw + 1e-6 * np.eye(self.D * self.N))[np.ix_(self.valid_idx, self.valid_idx)]

        self.y_vec = y_vec.reshape(-1, 1)[self.valid_idx,:]
        y_centered_w = self.y_vec - self.mw[self.valid_idx, :]

        self.L = cholesky(self.Kw)
        self.alpha = solve(self.L.T, solve(self.L, y_centered_w))

        return gamma, c

    def predict(self, X_test):
        K_s = (np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale)))[self.valid_idx, :]
        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + 1e-6 * np.eye(len(X_test) * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v

        mu = mu.reshape(self.D, -1).T
        var = np.diag(cov).reshape(self.D, -1).T

        return mu, var
    
    def loo_cv(self, length_scale, noise_var, A, k=1, fix_weights=True):        
        B = A @ A.T
        loo_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))

        noise_var_vec = np.kron(noise_var.reshape(-1,1), np.ones((self.N,1)))

        if fix_weights:
            loo_w01, loo_gradient_log_squared = self.init_w01, self.init_grad_log2
        else:
            loo_predictive_means, loo_predictive_variances = cross_channel_predictive(self.Y_train, self.mean, B, noise_var)
            loo_gamma = loo_predictive_means
            # loo_c = np.sqrt(loo_predictive_variances).reshape((-1,1), order='F')
            loo_c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - loo_gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))

            loo_w01, loo_gradient_log_squared = scaled_imq_weight(self.Y_train.T.reshape(-1,1), loo_gamma.reshape((-1,1), order='F'), loo_c)

        loo_Jw = np.diag((loo_w01**-2).flatten())
        loo_Kw = loo_K + np.kron(np.diag(noise_var), np.eye(self.N)) @ loo_Jw + 1e-6 * np.eye(self.D * self.N)
        loo_Kw_inv = np.linalg.inv(loo_Kw[np.ix_(self.valid_idx, self.valid_idx)])
        loo_Kw_inv_diag = np.diag(loo_Kw_inv).reshape(-1,1)
        z = self.y_vec - self.mean - (noise_var_vec * loo_gradient_log_squared)[self.valid_idx,:]

        # Compute LOO predictions
        loo_mean = z + self.mean - loo_Kw_inv @ z / loo_Kw_inv_diag
        loo_var = (1 / loo_Kw_inv_diag) - (noise_var_vec * (loo_w01**-2))[self.valid_idx] + (noise_var_vec.reshape(-1,1))[self.valid_idx,:]
        self.predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        return np.dot(self.predictive_log_prob.flatten(), self.init_w01[self.valid_idx, :].flatten()**k)
    
    def optimize_loo_cv(self, print_opt_param = False, print_iter_objective=False, k=1, init_cov=None, fix_weights=True, bounds=None):

        init_predictive_means, init_predictive_variance = cross_channel_predictive(self.Y_train, self.mean, init_cov, 0)
        init_gamma = init_predictive_means
        # init_c = np.sqrt(init_predictive_variance).reshape((-1,1), order='F')
        init_c = np.tile(np.array([np.nanquantile(np.abs(self.Y_train[:, d] - init_gamma[:, d]), 1 - self.epsilons[d], method='lower') for d in range(self.D)]), (self.N, 1))
        self.init_w01, self.init_grad_log2 = scaled_imq_weight(self.Y_train.T.reshape(-1,1), init_gamma.reshape((-1,1), order='F'), init_c.reshape((-1,1), order='F'))

        def objective(theta):
            length_scale = np.exp(theta)[0]
            noise_var = np.exp(theta)[1:self.D+1]
            A = theta[self.D+1:].reshape(self.D, -1)
            val = -self.loo_cv(length_scale, noise_var, A, k=k, fix_weights=fix_weights)
            if print_iter_objective:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale]),
            np.log(self.noise_var),
            self.A.reshape(-1)
        ))
        res = minimize(objective, initial_theta, method='L-BFGS-B', tol=1e-2,
                       bounds=[(np.log(1e-3), np.log(1e3))] +     # length_scale
                            [(np.log(b), np.log(b)) for b in bounds] +    # noise_var
                            [(np.log(1e-3), np.log(1e3))] * (self.A.shape[0] * self.A.shape[1])
        )

        self.length_scale = np.exp(res.x[0])
        self.noise_var = np.exp(res.x)[1:self.D+1]
        self.A = res.x[self.D+1:].reshape(self.D, -1)
        self.B = self.A @ self.A.T

        if print_opt_param:
            print(f"Optimized length_scale: {self.length_scale:.4f}")
            print(f"Optimized noise_var: {self.noise_var}")
            print(f"Optimized A: {self.A}")
            print(f"Optimized B: \n{self.B}")

        gamma, c = self.fit(self.X_train, self.Y_train, self.epsilons)
        return init_gamma, init_c, gamma, c
    
class MOGPRegressor_bounded_noises:
    def __init__(self, mean=0.0, length_scale=1.0, noise=None, A=None):
        self.D = A.shape[0]
        self.mean = mean
        self.length_scale = length_scale
        self.noise = noise
        self.noise_matrix = np.diag(noise)
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
        self.noise_matrix = np.diag(self.noise)

        y_vec = Y_train.T.flatten()
        self.mask = ~np.isnan(y_vec)
        self.valid_idx = np.where(self.mask)[0]
        y_vec = y_vec[self.mask].reshape(-1, 1)
        
        self.y_vec = y_vec
        full_K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        noise_K = np.kron(self.noise_matrix, np.eye(self.N))
        K = full_K + noise_K + 1e-6 * np.eye(self.D * self.N)

        self.K_noise = K[np.ix_(self.mask, self.mask)]
        y_centered = self.y_vec - self.mean

        self.L = cholesky(self.K_noise)
        self.alpha = solve(self.L.T, solve(self.L, y_centered))

    def predict(self, X_test):
        N_test = len(X_test)
        K_s = np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale))
        K_s = K_s[self.valid_idx, :] 

        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + \
               1e-6 * np.eye(N_test * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v
        var = np.maximum(0, np.diag(cov))

        mu = mu.reshape(self.D, -1).T
        var = var.reshape(self.D, -1).T

        return mu, var

    def log_marginal_likelihood(self, theta=None):
        if theta is not None:
            length_scale = np.exp(theta)[0]
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

    def optimize_hyperparameters(self, print_opt_param=False, print_iter_param=False, bounds=None):
        def objective(theta):
            val = -self.log_marginal_likelihood(theta)
            if print_iter_param:
                print(-val)
            return val

        initial_theta = np.concatenate((
            np.log([self.length_scale]),
            np.log(self.noise).ravel(),
            self.A.reshape(-1)
        ))

        # bounds = [
        #     (np.log(1e-2), np.log(1e2)),     # length_scale
        #     (np.log(1e-5), np.log(1.0)),     # noise
        # ] + [(np.log(1e-1), np.log(5))] * len(self.a)

        res = minimize(objective, initial_theta, method='L-BFGS-B', 
                       bounds=[(np.log(1e-3), np.log(1e3))] +     # length_scale
                            [(np.log(b), np.log(b)) for b in bounds] +    # noise_var
                            [(np.log(1e-3), np.log(1e3))] * (self.A.shape[0] * self.A.shape[1]))

        self.length_scale = np.exp(res.x[0])
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

    def loo_cv(self, length_scale, noise, A):
        B = A @ A.T
        loo_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))
        loo_K_noise = loo_K + np.kron(np.diag(noise), np.eye(self.N)) + 1e-6 * np.eye(self.D * self.N)
        loo_K_noise_inv = np.linalg.inv(loo_K_noise[np.ix_(self.mask, self.mask)])
        loo_K_noise_inv_diag = np.diag(loo_K_noise_inv).reshape(-1,1)

        # Compute LOO predictions
        loo_mean = self.y_vec - loo_K_noise_inv @ self.y_vec / loo_K_noise_inv_diag
        loo_var = 1 / loo_K_noise_inv_diag

        predictive_log_prob = -0.5 * np.log(loo_var) - 0.5 * (loo_mean - self.y_vec)**2/loo_var - 0.5 * np.log(np.pi * 2)

        return np.sum(predictive_log_prob)