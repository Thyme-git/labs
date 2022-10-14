from .tensor import Tensor
from .modules import Module


class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):

        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.

        for _, attr in vars(module).items():
            if isinstance(attr, Tensor):
                self._update_weight(attr)
            elif isinstance(attr, Module):
                self._step_module(attr)
            elif isinstance(attr, list) and len(attr) > 0 and isinstance(attr[0], Module):
                for m in attr:
                    self._step_module(m)
        # End of todo

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad
        

class SGD(Optim):

    def __init__(self, module, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum_dict = {}
        self.pre_v_dict = {}
        self.initialize(self.module)

    def initialize(self, module):
        for attr in vars(module).values():
            if isinstance(attr, Tensor):
                self.pre_v_dict[id(attr)] = 0
                self.momentum_dict[id(attr)] = 0
            elif isinstance(attr, Module):
                self.initialize(attr)
            elif isinstance(attr, list) and len(attr) > 0 and isinstance(attr[0], Module):
                for m in attr:
                    self.initialize(m)

    def _update_weight(self, tensor):

        # TODO Update the weight of tensor
        # in SGD manner.
        v = self.lr*tensor.grad + self.pre_v_dict[id(tensor)]*self.momentum_dict[id(tensor)]
        tensor -= v
        self.pre_v_dict[id(tensor)] = v
        # End of todo


class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)

        # TODO Initialize the attributes
        # of Adam optimizer.

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = 0
        self.v = 0
        self.t = 0

        # End of todo

    def _update_weight(self, tensor):

        # TODO Update the weight of
        # tensor in Adam manner.
        self.t += 1
        self.m = self.beta1*self.m + (1-self.beta1)*tensor.grad
        self.v = self.beta2*self.v + (1-self.beta2)*tensor.grad**2
        m_hat = self.m/(1-self.beta1**self.t)
        v_hat = self.v/(1-self.beta2**self.t)

        tensor.grad -= self.lr*m_hat/(v_hat**.5+self.epsilon)
        # End of todo