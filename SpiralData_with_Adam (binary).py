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

X, y = spiral_data(samples=100, classes=2)

y = y.reshape((-1, 1))

loss = BinaryCrossEntropy()
optimizer = AdamOptimizer(learning_rate_decay=5e-7)

dense1 = DenseLayer(2, 64, activation=ReLU, L2=5e-4, L1=0)
dense2 = DenseLayer(64, 1, activation=Sigmoid)

epochs = []
losses = []

plt.scatter(X[:, 0], X[:, 1], cmap="brg")
plt.show()

numEpochs = 10001

#Model training

for epoch in range(numEpochs):

    output1 = dense1.forward(X)
    output2 = dense2.forward(output1)
    loss.getLoss(output2, y)

    loss.backward()
    dense2.backward(loss.dinputs)
    dense1.backward(dense2.dinputs)

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
        
        predictions = (dense2.output > 0.5) * 1
        print("epoch:", epoch+1, "acc:", round(np.mean(predictions == y)*100, 2), "%", "loss:", loss.getLoss(output2, y) + L1lossW + L1lossB + L2lossW + L2lossB, "Reg loss:", L1lossW + L1lossB + L2lossW + L2lossB)
        epochs.append(epoch+1)
        losses.append(loss.getLoss(output2, y))


plt.plot(epochs, losses)
plt.show()
#acc: 95.67 % loss: 0.113375574

#Model validation
X_valid, y_valid = spiral_data(samples=100, classes=2)

y_valid = y_valid.reshape((-1, 1))

dense1.forward(X_valid)
dense2.forward(dense1.output)

predictions = (dense2.output > 0.5) * 1
print(round(np.mean(predictions == y) * 100, 2), "%", "loss:", loss.getLoss(dense2.output, y_valid))

#1000 samples, 512 neurons and 10,001 epochs
#82.0 % loss: 0.48695156 Dropout w regularization
#92.0 % loss: 0.28580382 Only regularization
#89.67 % loss: 0.39415303 Nothing
#88.0 % loss: 0.39361852 Only dropout

#1000 samples, 512 neurons and 20,001 epochs

#87.67 % loss: 0.335397 Only dropout