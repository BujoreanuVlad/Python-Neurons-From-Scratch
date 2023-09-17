import numpy as np
from Layers import Layer

class FeedForwardModel:

    def __init__(self, layers=[], loss=None, optimizer=None):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.trainable_layers = []

    def addLayer(self, layer: Layer):
        self.layers.append(layer)

    def set(self, loss=None, optimizer=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer

    def finalizeModel(self):

        for layer in self.layers:
            if hasattr(layer, "weights"):
                self.trainable_layers.append(layer)

    def train(self, X, y, *, epochs=1, print_every=1):

        self.finalizeModel()
        
        
    def predict(self, inputs):
        
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def getLoss(self):
        return self.loss
    
    def getOptimizer(self):
        return self.optimizer
    
    def forward(self):
        pass

    def backward(self):
        pass
