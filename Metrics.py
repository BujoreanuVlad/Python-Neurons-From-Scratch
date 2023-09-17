import numpy as np

def accuracy(predicted, actual):
    if len(actual.shape) == 1:
        return np.mean(np.argmax(predicted, axis=1) == actual)
    elif len(actual.shape) == 2:
        class_targets = np.argmax(actual)
        return np.mean(np.argmax(predicted, axis=1) == class_targets)

    return 0.

def numericAccuracy(predicted, actual, precision):

    difference = predicted - actual

    return np.mean(np.abs(difference) < precision)