import numpy as np
from ActivationFunctions import *
from Losses import L1Regularization, L2Regularization

#Abstract layer class
class Layer:
    def forward(self, inputs):
        return inputs
    
    def backward(self, gradients):
        return gradients

#Droupout layer
class DropoutLayer(Layer):

    def __init__(self, dropRate):
        self.dropRate = dropRate
    
    def forward(self, inputs):

        #Divide by 1 - dropRate to adjust for the fact that using less
        #neurons produces a smaller output on average
        self.bernoulliMask = np.random.binomial(1, 1-self.dropRate, inputs.shape) / (1 - self.dropRate)
        self.outputs = inputs * self.bernoulliMask

        return self.outputs

    def backward(self, gradient):
        
        self.dinputs = gradient * self.bernoulliMask
        return self.dinputs

#Abstract class for layers with trainable parameters (weights)
class TrainableLayer(Layer):
    def __init__(self) -> None:
        self.weights = np.empty(1)
        self.biases = np.empty(1)

class DenseLayer(TrainableLayer):

    def __init__(self, numInputs: int, numNeurons: int, activation=Activation, L1=0., L2=0.):
        #Initialize weights and biases
        self.weights = 0.1 * np.random.randn(numInputs, numNeurons)
        self.biases = np.zeros((1, numNeurons))
        self.activation = activation()
        self.numInputs = numInputs
        self.numNeurons = numNeurons
        self.L1 = L1
        self.L2 = L2
        self.L1Reg = L1Regularization(L1)
        self.L2Reg = L1Regularization(L2)

    def forward(self, inputs):
        #Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(self.output)
        return self.output

    def setWeights(self, new_weights):
        self.weights = new_weights

    def setBiases(self, new_biases):
        self.biases = new_biases
    
    def backward(self, gradient):

        self.activation.backward(gradient)
        self.L1Reg.backward(self)
        self.L2Reg.backward(self)
        gradient = self.activation.dinputs

        self.dweights = np.dot(self.inputs.T, gradient)
        if self.L1 != 0:
            self.dweights += self.L1Reg.dweights
        if self.L2 != 0:
            self.dweights += self.L2Reg.dweights

        self.dinputs = np.dot(gradient, self.weights.T)

        self.dbiases = np.sum(gradient, axis=0, keepdims=True)
        if self.L1 != 0:
            self.dbiases += self.L1Reg.dbiases
        if self.L2 != 0:
            self.dbiases += self.L2Reg.dbiases

class InputLayer(Layer):

    def forward(self, inputs):
        self.output = inputs
        return self.output