import tarfile
import numpy as np

from nn.tensor import tensor
from .modules import Module


class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.
        self.x = x
        return 1/(1+np.exp(-x))
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.

        f = 1/(1+np.exp(-self.x))    
        return dy*f*(1-f)
        # End of todo


class Tanh(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of tanh function.

        self.x = x
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.

        return dy*4*np.exp(2*self.x)/(np.exp(2*self.x)+1)**2
        # End of todo


class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.

        self.x = x
        return np.maximum(0, x)
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.
        
        dx = np.array(dy, copy=True)
        dx[self.x < 0] = 0
        return dx
        # End of todo


class Softmax(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of Softmax function.
        x -= np.max(x, axis=1, keepdims=True)
        expx = np.exp(x)
        return expx / np.sum(expx, axis=1, keepdims=True)
        # End of todo

    def backward(self, dy):

        # Omitted.
        ...


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.

        super(SoftmaxLoss, self).__call__(probs, targets)
        correct_probs = probs[range(probs.shape[0]), targets]
        return -1 * np.sum(correct_probs) / probs.shape[0]
        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.

        dx = self.probs
        N = self.targets.shape[0]

        dx[range(N), self.targets] -= 1
        return dx
        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.
        self.probs = probs
        self.targets = targets
        correct_probs = probs[range(probs.shape[0]), targets]
        self.value = -1 * np.sum(correct_probs) / probs.shape[0]
        return self
        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.
        dx = self.probs
        N = self.targets.shape[0]

        dx[range(N), self.targets] -= 1
        return dx
        # End of todo

