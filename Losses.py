import numpy as np
from ActivationFunctions import Softmax
from abc import abstractmethod
from Models import FeedForwardModel
from Layers import TrainableLayer

class Loss:

    @abstractmethod
    def forward(self, y_predicted, y_actual):
        return y_predicted

    def setTrainableLayers(self, model: FeedForwardModel) -> None:
        self.trainable_layers = model.trainable_layers

    def getLoss(self, output, y) -> float:
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class CategoricalCrossEntropy(Loss):

    def forward(self, y_predicted, y_actual):

        #Clipping the data so that we don't have log(0)
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1-1e-7)

        self.lastActual = y_actual
        self.lastInputs = y_predicted_clipped

        sample_losses = []

        if len(y_actual.shape) == 1:
            sample_losses = -np.log(y_predicted_clipped[range(len(y_actual)), y_actual])

        elif len(y_actual.shape) == 2:
            sample_losses = -np.log(np.sum(y_actual * y_predicted_clipped, axis=1))
        
        return sample_losses

    def backward(self):

        numSamples = len(self.lastInputs)

        #If the length of the shape is 1, then the y_actual are just the categories and we have
        #convert them to one-hot encoding
        if len(self.lastActual.shape) == 1:
            self.lastActual = np.eye(len(self.lastInputs[0]))[self.lastActual]
        
        self.dinputs = - self.lastActual / self.lastInputs
        self.dinputs /= numSamples

        return self.dinputs
    

class SoftmaxWithCategoricalCrossentropy(Loss):

    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, X, y_actual):

        self.activation.forward(X)
        self.activationOutputs = self.activation.lastOutput
        self.lastActual = y_actual
        return self.loss.getLoss(self.activationOutputs, y_actual)
        
    
    def backward(self):

        numSamples = len(self.activationOutputs)

        if len(self.lastActual.shape) == 2:
            self.lastActual = np.argmax(self.lastActual, axis=1)
        
        #Copy, so we can safely modify
        self.dinputs = self.activationOutputs.copy()
        #Calculate gradient (for each sample we take -1 from the value with the corresponding lastActual index)
        self.dinputs[range(numSamples), self.lastActual] -= 1
        #Normalize the gradient (the more samples we have the higher the total sum and could cause a gradient explosion)
        self.dinputs /= numSamples

class L1Regularization:

    def __init__(self, l: float):
        #l stands for lambda
        self.l = l
        
    def forward(self, layer):

        weights_regularization = np.sum(np.abs(layer.weights))
        biases_regularization = np.sum(np.abs(layer.biases))

        return self.l * weights_regularization, self.l * biases_regularization

    def backward(self, layer: TrainableLayer) -> None:

        self.dweights = np.ones_like(layer.weights)
        self.dweights[layer.weights < 0] = -1
        self.dweights *= self.l

        self.dbiases = np.ones_like(layer.biases)
        self.dbiases[layer.biases < 0] = -1
        self.dbiases *= self.l

class L2Regularization:

    def __init__(self, l: float):
        #l stands for lambda
        self.l = l

    def forward(self, layer: TrainableLayer):

        weights_regularization = np.sum(layer.weights ** 2)
        biases_regularization = np.sum(layer.biases ** 2)

        return self.l * weights_regularization, self.l * biases_regularization

    def backward(self, layer) -> None:

        self.dweights = layer.weights * 2 * self.l

        self.dbiases = layer.biases * 2 * self.l

class BinaryCrossEntropy(Loss):

    def forward(self, y_predicted, y_actual):

        #Clipping the data so that we don't have log(0)
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1-1e-7)
        self.lastActual = y_actual
        self.lastInputs = y_predicted_clipped

        sample_losses = -(y_actual * np.log(y_predicted_clipped) + (1 - y_actual) * np.log(1 - y_predicted_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
    
        return sample_losses

    def backward(self):

        numSamples = len(self.lastInputs)
        numOutputs = len(self.lastInputs[0])

        self.dinputs = -(self.lastActual / self.lastInputs) + (1-self.lastActual) / (1-self.lastInputs) / numOutputs
        self.dinputs /= numSamples

        return self.dinputs
    
class MeanSquaredError(Loss):

    def forward(self, y_predicted, y_actual):

        self.predicted = y_predicted
        self.actual = y_actual

        return np.mean(np.square(y_predicted - y_actual), axis=-1)
    
    def backward(self):

        #Number of outputs per sample
        num_outputs = len(self.predicted[0])
        num_samples = len(self.predicted)

        self.dinputs = -2 * (self.actual - self.predicted) / num_outputs
        self.dinputs /= num_samples

        return self.dinputs

class MeanAbsoluteError(Loss):

    def forward(self, y_predicted, y_actual):

        self.predicted = y_predicted
        self.actual = y_actual

        return np.mean(np.abs(y_predicted - y_actual), axis=-1)
    
    def backward(self):

        #Number of outputs per sample
        num_outputs = len(self.predicted[0])
        num_samples = len(self.predicted)

        self.dinputs = np.sign(self.actual - self.predicted) / num_outputs
        self.dinputs /= num_samples

        return self.dinputs
    