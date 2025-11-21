import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        self.A = A
        self.originShape = A.shape

        if len(self.originShape) > 1:
            size = int(np.prod(self.originShape[:-1]))
        else:
            size = 1

        in_features = self.W.shape[1]
        A_flat = A.reshape(size, in_features)
        Z_flat = A_flat @ self.W.T + self.b
        out_features = self.W.shape[0]
        Z = Z_flat.reshape(*self.originShape[:-1], out_features)
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        shape = self.originShape

        if len(shape) > 1:
            size = int(np.prod(shape[:-1]))
        else:
            size = 1
        out_features = dLdZ.shape[-1]
        flattened_dLdZ = dLdZ.reshape(size, out_features)
        flattenedA = self.A.reshape(size, shape[-1])

        flattened_dLdA = flattened_dLdZ @ self.W
        self.dLdW = flattened_dLdZ.T @ flattenedA
        self.dLdb = flattened_dLdZ.sum(axis=0)

        self.dLdA = flattened_dLdA.reshape(*shape)
        return self.dLdA
    
