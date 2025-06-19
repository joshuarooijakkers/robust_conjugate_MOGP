import numpy as np

class ConstantMean:
    def __init__(self, constant=0.0):
        self.constant = constant

    def __call__(self, X):
        """Compute the constant mean vector."""
        return np.full((X.shape[0], 1), self.constant)

class SineMean:
    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def __call__(self, X):
        """
        Compute the sine mean vector.
        Applies sine to the first dimension of X.
        """
        return self.amplitude * np.sin(self.frequency * X[:, [0]] + self.phase)

class RBFKernel:
    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, X1, X2):
        """Compute the RBF kernel matrix."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.variance * np.exp(-0.5 / self.lengthscale**2 * sqdist)