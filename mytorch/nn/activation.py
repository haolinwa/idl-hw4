import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*)I to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        self.Z_shape = Z.shape
        dim = self.dim if self.dim >= 0 else Z.ndim + self.dim

        Z_moved = np.moveaxis(Z, dim, -1)
        orig_shape = Z_moved.shape
        N = int(np.prod(orig_shape[:-1]))
        D = orig_shape[-1]

        Z_flat = Z_moved.reshape(N, D)
        Z_flat = Z_flat - Z_flat.max(axis=1, keepdims=True)
        expZ = np.exp(Z_flat)
        A_flat = expZ / expZ.sum(axis=1, keepdims=True)

        A_moved = A_flat.reshape(orig_shape)
        self.A = np.moveaxis(A_moved, -1, dim)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        dim = self.dim if self.dim >= 0 else dLdA.ndim + self.dim

        A = self.A
        A_moved = np.moveaxis(A, dim, -1)
        dLdA_moved = np.moveaxis(dLdA, dim, -1)

        orig_shape = A_moved.shape
        N = int(np.prod(orig_shape[:-1]))
        D = orig_shape[-1]

        A_flat = A_moved.reshape(N, D)
        dLdA_flat = dLdA_moved.reshape(N, D)

        dot = np.sum(dLdA_flat * A_flat, axis=1, keepdims=True)
        dLdZ_flat = A_flat * (dLdA_flat - dot)

        dLdZ_moved = dLdZ_flat.reshape(orig_shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim)

        self.dLdZ = dLdZ
        return dLdZ
 

    