from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
from DenseLayer import DenseLayer
from DropoutLayer import DropoutLayer
from ActivationFunctions import *
from Losses import *
from Models import FeedForwardModel
from Metrics import *
from Optimizers import *

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)

loss = CategoricalCrossEntropy()
lossAct = SoftmaxWithCategoricalCrossentropy()
optimizer = AdamOptimizer(learning_rate=0.02, learning_rate_decay=5e-7, beta1=0.99, beta2=0.999)

dense1 = DenseLayer(2, 512, activation=ReLU, L2=0, L1=0)
drop1 = DropoutLayer(0.1)
dense2 = DenseLayer(512, 3, activation=Linear)

epochs = []
losses = []

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()

numEpochs = 20001

#Model training

for epoch in range(numEpochs):

    output1 = dense1.forward(X)
    drop1.forward(output1)
    output2 = dense2.forward(drop1.outputs)
    lossAct.forward(output2, y)
    loss.getLoss(output2, y)

    lossAct.backward()
    dense2.backward(lossAct.dinputs)
    drop1.backward(dense2.dinputs)
    dense1.backward(drop1.dinputs)

    optimizer.preOptimize()
    optimizer.optimize(dense1)
    optimizer.optimize(dense2)
    optimizer.postOptimize()

    if epoch % 100 == 0:
        
        L1lossW = L1lossB = L2lossW = L2lossB = 0
        _w, _b = dense1.L1Reg.forward(dense1)
        L1lossW += _w
        L1lossB += _b
        _w, _b = dense1.L2Reg.forward(dense1)
        L2lossW += _w
        L2lossB += _b

        _w, _b = dense2.L1Reg.forward(dense2)
        L1lossW += _w
        L1lossB += _b
        _w, _b = dense2.L2Reg.forward(dense2)
        L2lossW += _w
        L2lossB += _b
        
        print("epoch:", epoch+1, "acc:", round(accuracy(lossAct.activationOutputs, y)*100, 2), "%", "loss:", lossAct.getLoss(output2, y) + L1lossW + L1lossB + L2lossW + L2lossB, "Reg loss:", L1lossW + L1lossB + L2lossW + L2lossB)
        epochs.append(epoch+1)
        losses.append(lossAct.getLoss(output2, y))


plt.plot(epochs, losses)
plt.show()
#acc: 95.67 % loss: 0.113375574

#Model validation
X_valid, y_valid = spiral_data(samples=100, classes=3)

dense1.forward(X_valid)
dense2.forward(dense1.output)
loss = lossAct.forward(dense2.output, y_valid)

print(round(accuracy(lossAct.activationOutputs, y_valid) * 100, 2), "%", "loss:", loss)

#1000 samples, 512 neurons and 10,001 epochs
#82.0 % loss: 0.48695156 Dropout w regularization
#92.0 % loss: 0.28580382 Only regularization
#89.67 % loss: 0.39415303 Nothing
#88.0 % loss: 0.39361852 Only dropout

#1000 samples, 512 neurons and 20,001 epochs

#87.67 % loss: 0.335397 Only dropout