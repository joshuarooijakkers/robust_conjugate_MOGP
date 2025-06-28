import numpy as np
from rcgp.kernels import RBFKernel
from scipy.optimize import minimize
from numpy.linalg import cholesky, solve

def imq_kernel(y, x, beta, c):
    imq = beta * (1 + ((y-x)**2)/(c**2))**(-0.5)
    gradient_log_squared = 2 * (x - y)/(c**2) * (1+(y-x)**2/(c**2))**-1
    return imq, gradient_log_squared

class MOGPRegressor:
    def __init__(self, n_outputs, mean=0.0, length_scale=1.0, noise=1e-2, A=None):
        self.D = n_outputs
        self.mean = mean
        self.length_scale = length_scale
        self.noise = noise
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
        mask = ~np.isnan(y_vec)
        self.valid_idx = np.where(mask)[0]
        y_vec = y_vec[mask].reshape(-1, 1)
        
        self.y_vec = y_vec
        # Kernel matrix for all outputs
        full_K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        noise_K = np.kron(self.noise * np.eye(self.D), np.eye(self.N))
        K = full_K + noise_K + 1e-6 * np.eye(self.D * self.N)

        # Subset kernel matrix and solve only for valid indices
        self.K_noise = K[np.ix_(mask, mask)]
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
            noise = np.exp(theta)[1]
            A = theta[2:].reshape(self.D, -1)
            B = A @ A.T
            full_K = np.kron(B, self.rbf_kernel(self.X_train, self.X_train, length_scale))
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

    def optimize_hyperparameters(self):
        def objective(theta):
            return -self.log_marginal_likelihood(theta)

        initial_theta = np.concatenate((
            np.log([self.length_scale, self.noise]),
            self.A.reshape(-1)
        ))

        # bounds = [
        #     (np.log(1e-2), np.log(1e2)),     # length_scale
        #     (np.log(1e-5), np.log(1.0)),     # noise
        # ] + [(np.log(1e-1), np.log(5))] * len(self.a)

        res = minimize(objective, initial_theta, method='L-BFGS-B')

        self.length_scale = np.exp(res.x[0])
        self.noise = np.exp(res.x[1])
        self.A = res.x[2:].reshape(self.D,-1)
        self.B = self.A @ self.A.T

        print(f"Optimized length_scale: {self.length_scale:.4f}, noise: {self.noise:.6f}")
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

        self.y_vec = Y_train.T.flatten().reshape(self.N * self.D, 1)

        beta = (self.noise / 2)**0.5

        c = np.quantile(self.Y_train, 1 - self.epsilon, axis=0).reshape(-1,1)

        w, gradient_log_squared = imq_kernel(self.y_vec, self.mean, beta, np.kron(c, np.ones((self.N, 1))))

        self.mw = self.mean + self.noise * gradient_log_squared
        self.Jw = (self.noise/2) * np.diag((w**-2).flatten())

        self.K = np.kron(self.B, self.rbf_kernel(X_train, X_train, self.length_scale))
        self.Kw = self.K + np.kron(self.noise * np.eye(self.D), np.eye(self.N)) @ self.Jw + 1e-6 * np.eye(self.D * self.N)

        y_centered_w = self.y_vec - self.mw

        self.L = cholesky(self.Kw)
        self.alpha = solve(self.L.T, solve(self.L, y_centered_w))

    def predict(self, X_test):
        K_s = np.kron(self.B, self.rbf_kernel(self.X_train, X_test, self.length_scale))
        K_ss = np.kron(self.B, self.rbf_kernel(X_test, X_test, self.length_scale)) + 1e-6 * np.eye(len(X_test) * self.D)

        mu = K_s.T @ self.alpha + self.mean
        v = solve(self.L, K_s)
        cov = K_ss - v.T @ v
        std = np.sqrt(np.diag(cov))

        mu = mu.reshape(self.D, -1).T
        std = np.sqrt(np.diag(cov)).reshape(self.D, -1).T

        return mu, std
