from Layers import *
from ActivationFunctions import *
from Losses import MeanSquaredError, MeanAbsoluteError
from Optimizers import AdamOptimizer
import numpy as np
import nnfs
from nnfs.datasets import sine_data
from Metrics import numericAccuracy
import matplotlib.pylab as plt
from Models import FeedForwardModel

nnfs.init()

X, y = sine_data()

model = FeedForwardModel([
    DenseLayer(1, 64, activation=ReLU),
    DenseLayer(64, 64, activation=ReLU),
    DenseLayer(64, 1)
])

loss_function = MeanSquaredError()
optimizer = AdamOptimizer(learning_rate=0.005, learning_rate_decay=1e-3)

num_epochs = 10000

for epoch in range(num_epochs):

    dense1.forward(X)
    dense2.forward(dense1.output)
    dense3.forward(dense2.output)
    loss_function.getLoss(dense3.output, y)

    loss_function.backward()
    dense3.backward(loss_function.dinputs)
    dense2.backward(dense3.dinputs)
    dense1.backward(dense2.dinputs)

    optimizer.preOptimize()
    optimizer.optimize(dense1)
    optimizer.optimize(dense2)
    optimizer.optimize(dense3)
    optimizer.postOptimize()

    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss_function.getLoss(dense3.output, y), "Accuracy:", numericAccuracy(dense3.output, y, np.std(y)/250))

plt.plot(dense3.output)
plt.plot(y)
plt.show()