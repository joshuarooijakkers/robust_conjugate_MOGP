import numpy as np
# from rcgp.kernels import RBFKernel

def imq_kernel(y, x, beta, c):
    imq = beta * (1 + ((y-x)**2)/(c**2))**-0.5
    gradient_log_squared = 2 * (x - y)/(c**2) * (1+(y-x)**2/(c**2))**-1
    return imq, gradient_log_squared

class GPRegressor:
    def __init__(self, mean, kernel, noise=1e-5):
        self.mean = mean
        self.kernel = kernel
        self.noise = noise
        self.is_fitted = False

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train.reshape(-1, 1)
        self.n = len(X_train)

        self.m = self.mean(X_train)
        self.K = self.kernel(X_train, X_train)

        self.K_noise = self.K + self.noise * np.eye(len(self.X_train)) + 1e-6
        self.K_noise_inv = np.linalg.inv(self.K + self.noise * np.eye(len(self.X_train)))

        self.is_fitted = True

    def predict(self, X_test):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")

        K_star = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        mu = self.mean(X_test) + K_star.T @ self.K_noise_inv @ (self.y_train - self.m)
        cov = K_ss - K_star.T @ self.K_noise_inv @ K_star

        return mu, cov

    def negative_mll(self, params):
        lengthscale, variance, noise = np.exp(params)
        kernel = RBFKernel(lengthscale, variance)
        K = kernel(self.X_train, self.X_train)
        K_noise = K + noise * np.eye(len(self.X_train))
        m = self.mean(self.X_train)

        try:
            K_inv = np.linalg.inv(K_noise)
            diff = self.y_train - m
            log_det = np.linalg.slogdet(K_noise)[1]  # More stable than np.log(det)

            mll = -0.5 * diff.T @ K_inv @ diff - 0.5 * log_det - 0.5 * self.n * np.log(2 * np.pi)
            mll = mll.item()

            print(f"[INFO] lengthscale={lengthscale:.3f}, variance={variance:.3f}, noise={noise:.6f}, -MLL={-mll:.3f}")
        except np.linalg.LinAlgError as e:
            print(f"[ERROR] LinAlg error: {e}")
            print(f"[DEBUG] Params: lengthscale={lengthscale}, variance={variance}, noise={noise}")
            return 1e6  # penalize bad params

        return -mll

    def approximate_gradient(self, params, epsilon=1e-5):
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_eps_plus = params.copy()
            params_eps_minus = params.copy()
            params_eps_plus[i] += epsilon
            params_eps_minus[i] -= epsilon
            f_plus = self.negative_mll(params_eps_plus)
            f_minus = self.negative_mll(params_eps_minus)
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        return grad

    def optimize_hyperparameters(self, initial_params=np.log([1.0, 1.0, 0.1]),
                                          learning_rate=0.1,
                                          max_iters=100,
                                          tol=1e-4):
        params = initial_params.copy()
        for i in range(max_iters):
            grad = self.approximate_gradient(params)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tol:
                print(f"Converged at iteration {i}")
                break

            params -= learning_rate * grad

        lengthscale, variance, noise = np.exp(params)
        self.kernel = RBFKernel(lengthscale, variance)
        self.noise = noise
        print(f"Optimized params: lengthscale={lengthscale:.3f}, variance={variance:.3f}, noise={noise:.6f}")
        self.fit(self.X_train, self.y_train)

class RCGPRegressor:
    def __init__(self, mean, kernel, noise=1e-5):
        self.mean = mean
        self.kernel = kernel
        self.noise = noise
        self.is_fitted = False

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train.reshape(-1, 1)
        self.n = len(X_train)

        self.m = self.mean(X_train)
        self.K = self.kernel(X_train, X_train)

        beta = (self.noise / 2)**0.5
        c = 1

        w, gradient_log_squared = imq_kernel(self.y_train, self.m, beta, c)

        self.mw = self.m + self.noise * gradient_log_squared
        self.Jw = (self.noise/2) * np.diag((w**-2).flatten())

        self.is_fitted = True

    def predict(self, X_test):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")

        K_star = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        K_noise_w = self.K + self.noise * self.Jw
        K_noise_w_inv = np.linalg.inv(K_noise_w)

        mu = self.mean(X_test) + K_star.T @ K_noise_w_inv @ (self.y_train - self.mw)
        cov = K_ss - K_star.T @ K_noise_w_inv @ K_star

        return mu, cov
