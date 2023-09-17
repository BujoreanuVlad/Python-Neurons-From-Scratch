import numpy as np

class Activation:
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        self.dinputs = gradients
        return gradients

class Step(Activation):

    def forward(self, X):
        for i in range(len(X)):
            if X[i] > 0:
                X[i] = 1
            else:
                X[i] = 0
        return X
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        return np.zeros_like(gradients)

class Sigmoid(Activation):

    def forward(self, X):
        self.lastOutput =  1 / (1 + np.exp(-X))
        return self.lastOutput

    def backward(self, gradient):
        self.dinputs = gradient * (1-self.lastOutput) * self.lastOutput
        return self.dinputs


class ReLU(Activation):
    
    def forward(self, X):
        self.inputs = X
        return np.maximum(0, X)

    def backward(self, dvalues):
        
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Linear(Activation):

    def forward(self, X):
        return X

    def backward(self, gradient):
        self.dinputs = gradient

class Softmax(Activation):

    def forward(self, X):
        #We substract the max value so that all our values are <= 0 to prevent
        #overflow when we exponentiate
        self.lastInputs = X
        X = X - np.max(X, axis=1, keepdims=True)
        X = np.exp(X)
        X = X / np.sum(X, axis=1, keepdims=True)
        self.lastOutput = X
        return X
    
    def backward(self, dvalues):
        
        self.dinputs = np.empty_like(dvalues)

        for index, (oneOutput, previousGradient) in enumerate(zip(self.lastOutput, dvalues)):

            #Transform row vector to column vector
            oneOutput = oneOutput.reshape((-1, 1))
            #On the diagonal it will be: S1 - S1 * S1 = S1 * (1 - S1)
            #and otherwise: 0 - S1 * S2 = -S1 * S2
            JacobianMatrix = np.diagflat(oneOutput) - np.dot(oneOutput, oneOutput.T)

            self.dinputs[index] = np.dot(JacobianMatrix, previousGradient)

