from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
from DenseLayer import DenseLayer
from ActivationFunctions import *
from Losses import *
from Models import FeedForwardModel
from Metrics import *
from Optimizers import *

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

loss = CategoricalCrossEntropy()
lossAct = SoftmaxWithCategoricalCrossentropy()
optimizer = AdaGradOptimizer(learning_rate_decay=1e-4)

dense1 = DenseLayer(2, 64, activation=ReLU)
dense2 = DenseLayer(64, 3, activation=Linear)

epochs = []
losses = []

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()

numEpochs = 10001

for epoch in range(numEpochs):

    output1 = dense1.forward(X)
    output2 = dense2.forward(output1)
    lossAct.forward(output2, y)
    #loss.getLoss(output2, y)

    lossAct.backward()
    dense2.backward(lossAct.dinputs)
    dense1.backward(dense2.dinputs)

    optimizer.preOptimize()
    optimizer.optimize(dense1)
    optimizer.optimize(dense2)
    optimizer.postOptimize()

    if epoch % 100 == 0:
        print("epoch:", epoch+1, "acc:", round(accuracy(output2, y)*100, 2), "%", "loss:", lossAct.getLoss(output2, y))
        epochs.append(epoch+1)
        losses.append(lossAct.getLoss(output2, y))

        #print(output2[:5])


plt.plot(epochs, losses)
plt.show()
#acc: 95.67 % loss: 0.113375574