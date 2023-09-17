import numpy as np

class Optimizer:

    def optimize(self, X, y):
        pass

class RandomWeightOptimizer(Optimizer):
    def __init__(self, model):
        self.model = model
    
    def optimize(self, X, y):

        for i in range(len(self.model.layers)):
            shape = (self.model.layers[i].numInputs, self.model.layers[i].numNeurons)
            self.model.layers[i].setWeights(0.05 * np.random.randn(shape[0], shape[1]))
            self.model.layers[i].setBiases(0.05 * np.random.randn(1, shape[1]))

class RandomAdjustmentOptimizer(Optimizer):
    def __init__(self, model, loss, alpha=0.05):
        self.model = model
        self.loss = loss
        self.alpha = alpha
    
    def optimize(self, X, y):

        current_loss = self.loss.getLoss(self.model.predict(X), y)

        copy_layers = []

        for layer in self.model.layers:
            copy_layers.append(layer)

        #Adjust the weights and biases in each layer randomly
        for i in range(len(self.model.layers)):
            shape = (self.model.layers[i].numInputs, self.model.layers[i].numNeurons)
            self.model.layers[i].setWeights(self.model.layers[i].weights + self.alpha * np.random.randn(shape[0], shape[1]))
            self.model.layers[i].setBiases(self.model.layers[i].biases + self.alpha * np.random.randn(1, shape[1]))

        new_loss = self.loss.getLoss(self.model.predict(X), y)

        #If new loss isn't better, reset the layer's weights and biases
        if new_loss > current_loss:
            for i in range(len(self.model.layers)):
                self.model.layers[i] = copy_layers[i]

    
    def setAlpha(self, alpha):
        self.alpha = alpha

class SGDOptimizer(Optimizer):

    def __init__(self, learning_rate=1., learning_rate_decay=0., momentum=0.):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = learning_rate_decay
        self.iteration = 0
        self.momentum = momentum

    def preOptimize(self):
        self.current_learning_rate = self.starting_learning_rate / (1. + self.iteration * self.decay)

    def optimize(self, layer):

        if self.momentum:
            
            if not hasattr(layer, "weights_momentum"):
                layer.weights_momentum = np.zeros_like(layer.weights)
                layer.biases_momentum = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weights_momentum - self.current_learning_rate * layer.dweights
            biases_updates = self.momentum * layer.biases_momentum - self.current_learning_rate * layer.dbiases

            layer.weights_momentum = weight_updates
            layer.biases_momentum = biases_updates

            layer.weights += weight_updates
            layer.biases += biases_updates

        else:

            weight_updates = -self.current_learning_rate * layer.dweights
            biases_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += biases_updates

    def postOptimize(self):
        self.iteration += 1

class AdaGradOptimizer(Optimizer):

    def __init__(self, learning_rate=1., learning_rate_decay=0., epsilon=1e-7):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = learning_rate_decay
        self.iteration = 0
        self.epsilon = epsilon

    def preOptimize(self):
        self.current_learning_rate = self.starting_learning_rate / (1. + self.iteration * self.decay)

    def optimize(self, layer):
           
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        weight_updates =  -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        biases_updates = -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights += weight_updates
        layer.biases += biases_updates

    def postOptimize(self):
        self.iteration += 1

class RMSPropOptimizer(Optimizer):

    def __init__(self, learning_rate=0.001, learning_rate_decay=0., epsilon=1e-7, rho=0.9):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = learning_rate_decay
        self.iteration = 0
        self.epsilon = epsilon
        self.rho = rho

    def preOptimize(self):
        self.current_learning_rate = self.starting_learning_rate / (1. + self.iteration * self.decay)

    def optimize(self, layer):
           
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        weight_updates =  -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        biases_updates = -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights += weight_updates
        layer.biases += biases_updates

    def postOptimize(self):
        self.iteration += 1

class AdamOptimizer(Optimizer):

    def __init__(self, learning_rate=0.001, learning_rate_decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.starting_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = learning_rate_decay
        self.iteration = 0
        self.epsilon = epsilon
        #beta1 plays the role the rho parameter in RMSprop did, but for the momentums
        self.beta1 = beta1
        #beta2 is for the same as beta1 but for the cache
        self.beta2 = beta2

    def preOptimize(self):
        self.current_learning_rate = self.starting_learning_rate / (1. + self.iteration * self.decay)

    def optimize(self, layer):
           
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
        
        #Calculating the momentums
        #Adding current gradients to the momentum, RMSprop style
        layer.weight_momentum = self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.dweights
        layer.bias_momentum = self.beta1 * layer.bias_momentum + (1- self.beta1) * layer.dbiases

        #Bias correction mechanism
        corrected_weight_momentum = layer.weight_momentum / (1 - self.beta1 ** (self.iteration+1))
        corrected_bias_momentum = layer.bias_momentum / (1 - self.beta1 ** (self.iteration+1))

        #Calculating the caches
        #Adding current gradients to the cache, RMSprop style
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases ** 2

        #Bias correction mechanism
        corrected_weight_cache = layer.weight_cache / (1 - self.beta2 ** (self.iteration+1))
        corrected_bias_cache = layer.bias_cache / (1 - self.beta2 ** (self.iteration+1))

        weight_updates =  -self.current_learning_rate * corrected_weight_momentum / (np.sqrt(corrected_weight_cache) + self.epsilon)
        biases_updates = -self.current_learning_rate * corrected_bias_momentum / (np.sqrt(corrected_bias_cache) + self.epsilon)

        layer.weights += weight_updates
        layer.biases += biases_updates

    def postOptimize(self):
        self.iteration += 1