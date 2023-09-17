from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
from DenseLayer import DenseLayer
from ActivationFunctions import *
from Losses import *
from Models import FeedForwardModel
from Metrics import *

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

#Dense layer with 2 input features and 3 neurons
dense1 = DenseLayer(2, 3, activation=ReLU)
output1 = dense1.forward(X)

dense2 = DenseLayer(3, 3, activation=ReLU)
output2 = dense2.forward(output1)

output_layer = DenseLayer(3, 3, activation=Softmax)
output = output_layer.forward(output2)
print(output[:5])

loss = CategoricalCrossEntropy()
print(loss.getLoss(output, y))
print(accuracy(output, y))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()

