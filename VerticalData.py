from nnfs.datasets import vertical_data
import nnfs
import matplotlib.pyplot as plt
from DenseLayer import DenseLayer
from ActivationFunctions import *
from Losses import *
from Models import FeedForwardModel
from Metrics import *
from Optimizers import *

nnfs.init()

X, y = vertical_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

model = FeedForwardModel([
    DenseLayer(2, 3, activation=ReLU),
    DenseLayer(3, 3, activation=Softmax)
])

loss = CategoricalCrossEntropy()
#optimizer = RandomWeightOptimizer(model)
optimizer = RandomAdjustmentOptimizer(model, loss)

current_loss = 9999

frustration_meter = 0
burn_out = 0

for i in range(100_000):

    predictions = model.predict(X)
    if loss.getLoss(predictions, y) < current_loss:
        current_loss = loss.getLoss(predictions, y)
        print("Current epoch:", i+1)
        print("Current loss:", loss.getLoss(predictions, y))
        print("Current accuracy: ", round(accuracy(predictions, y) * 100, 2), "%", sep='')
        optimizer.setAlpha(optimizer.alpha * 0.9)
        frustration_meter = 0
        burn_out = 0
        print(optimizer.alpha)
    else:
        frustration_meter += 1
    
    if frustration_meter >= 100 + i // 100:
        optimizer.setAlpha(optimizer.alpha * 1.1)
        burn_out += 1
        frustration_meter = 0

    #For the current design, the burnout mechanic has no effect
    if burn_out >= 100:
        print("Burnout:", optimizer.alpha)
        optimizer.setAlpha(optimizer.alpha * 0.3)
        burn_out = 0

    optimizer.optimize(X, y)

#Best result so far:
#Current epoch: 3375
#Current loss: 0.43980867
#Current accuracy: 86.33%