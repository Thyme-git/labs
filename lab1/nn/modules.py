from ast import BitAnd
from operator import length_hint
from time import sleep
import numpy as np
from itertools import product
from . import tensor
from .im2row import *


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # w[0] for bias and w[1:] for weight
        self.w = tensor.tensor((in_length + 1, out_length))

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.
        
        # out = x*w + b
        self.x = x
        out = x.dot(self.w[1:]) + self.w[0]
        return np.array(out)
        # End of todo

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.
        
        # dw = x.T*dy
        # dx = dy*w.T 
        # db = dy sum up by column
        self.w.grad = np.vstack((np.sum(dy, axis=0), (self.x.T).dot(dy)))
        dx = dy.dot(self.w[1:].T)
        return np.array(dx)
        # End of todo


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.

        self.gamma = tensor.ones(length)
        self.beta = tensor.zeros(length)
        self.moving_mean = np.zeros(length)
        self.moving_variance = np.ones(length)
        self.momentum = momentum
        self.length = length
        self.epsilon = 1e-8
        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.

        mean = x.mean(axis = 0)
        self.var = np.array(x.std(axis = 0))**2
        self.x_hat = np.array((x-mean)/(self.var+self.epsilon)**.5)
        y_hat = self.gamma*self.x_hat + self.beta
        self.moving_mean = np.array(self.momentum*mean + (1-self.momentum)*self.moving_mean)
        self.moving_variance = np.array(self.momentum*self.var + (1-self.momentum)*self.moving_variance)
        return np.array(y_hat)
        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.
        
        N = dy.shape[0]
        self.gamma.grad = np.sum(np.multiply(self.x_hat, dy), axis = 0)
        self.beta.grad = np.sum(dy, axis = 0)
        return np.array(np.multiply(dy, self.gamma/self.var**.5*(1-N)))
        # End of todo


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=True):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = channels
        self.kernel_size = 3
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.n_kernel = channels
        self.kernel = tensor.ones((channels, in_channels, kernel_size, kernel_size))
        if self.use_bias:
            self.bias = tensor.zeros(channels)
        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.
        # x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        # h_out = int((x.shape[2]+2*self.padding-self.kernel.shape[2])/self.stride)+1
        # w_out = int((x.shape[3]+2*self.padding-self.kernel.shape[3])/self.stride)+1
        # out = tensor.from_array(np.zeros((x.shape[0], self.n_kernel, h_out, w_out)))
        
        # for b in range(x.shape[0]):
        #     for c in range(self.n_kernel):
        #         for i in range(0, out.shape[2]):
        #             for j in range(0, out.shape[3]):
        #                 out[b, c, i, j] = np.sum(np.multiply(self.kernel[c], x[b:b+1, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]))
        
        # for c in range(self.n_kernel):
        #     out[:, c, :, :] += self.bias[c]

        # self.x = x
        # self.kernel.grad = tensor.zeros(self.kernel.shape)
        # self.bias.grad = tensor.zeros(self.bias.shape)
        # return out
        # End of todo

        #ver2
        b, c, h, w = x.shape
        self.input_shape = x.shape
        
        out_h = int((h - self.kernel_size + 2 * self.padding) / self.stride + 1)
        out_w = int((w - self.kernel_size + 2 * self.padding) / self.stride + 1)

        self.x = im2row_indices(x, self.kernel_size, self.kernel_size, stride = self.stride, padding = self.padding)
        k = np.matrix(kernel2row(self.kernel)).T
        out = np.array(self.x.dot(k))

        if self.use_bias:
            out += self.bias

        return conv_row2output(out, b, out_h, out_w)

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.
        # dx = tensor.zeros(self.x.shape)
        # for b in range(self.x.shape[0]):
        #     for c in range(self.n_kernel):
        #         for i in range(0, dy.shape[2]):
        #             for j in range(0, dy.shape[3]):
        #                 dx[b:b+1, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += self.kernel[c]*dy[b, c, i, j]
        #                 self.kernel.grad[c] += (dy[b:b+1, c:c+1, i:i+1, j:j+1]*dx[b:b+1, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size])[0]

        # return dx

        # ver2
        input_shape, k = self.input_shape, np.matrix(kernel2row(self.kernel)).T

        dy = conv_output2row(dy)
        self.kernel.grad = self.x.T.dot(dy)
        self.kernel.grad = row2kernel(self.kernel.grad, self.kernel.shape)
        if self.use_bias:
            self.bias.grad = np.sum(dy, axis=0, keepdims=True).reshape(-1)

        dx = np.array(dy.dot(k.T))
        dx = row2im_indices(dx, input_shape, field_height=self.kernel_size,
                            field_width=self.kernel_size, stride=self.stride, padding=self.padding)
        return dx

        # End of todo


class Conv2d_im2col(Conv2d):

    def forward(self, x):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.
        # print(x.shape)
        # print((self.kernel_size, self.stride, self.padding, self.n_kernel))
        # out = im2row_indices(x, self.kernel_size, self.kernel_size, stride = self.stride, padding = self.padding)
        # print(out.shape)
        # return out
        b, c, h, w = x.shape
        self.input_shape = x.shape
        
        out_h = int((h - self.kernel_size + 2 * self.padding) / self.stride + 1)
        out_w = int((w - self.kernel_size + 2 * self.padding) / self.stride + 1)

        self.x = im2row_indices(x, self.kernel_size, self.kernel_size, stride = self.stride, padding = self.padding)
        k = np.matrix(kernel2row(self.kernel)).T
        out = np.array(self.x.dot(k))

        if self.use_bias:
            out += self.bias

        return conv_row2output(out, b, out_h, out_w)
        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.
        B, C, H, W = x.shape
        self.input_shape = x.shape
        H_out = int((H - self.kernel_size) / self.stride + 1)
        W_out = int((W - self.kernel_size) / self.stride + 1)

        x = pool2row_indices(x, self.kernel_size, self.kernel_size, stride=self.stride)
        out = np.mean(x, axis=1)

        self.x_shape = x.shape

        return pool_row2output(out, B, H_out, W_out)
        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.

        dy = pool_output2row(dy)
        dx = np.zeros(self.x_shape)
        dy = dy.reshape((dy.shape[0], 1))
        dx[range(self.x_shape[0])] = dy
        dx /= self.x_shape[1]

        return row2pool_indices(dx, self.input_shape, field_height=self.kernel_size, field_width=self.kernel_size,
                                stride=self.stride)

        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.

        B, C, H, W = x.shape
        self.input_shape = x.shape
        H_out = int((H - self.kernel_size) / self.stride + 1)
        W_out = int((W - self.kernel_size) / self.stride + 1)

        x = pool2row_indices(x, self.kernel_size, self.kernel_size, stride=self.stride)
        out = np.max(x, axis=1)

        self.arg_out = np.argmax(x, axis=1)
        self.x_shape = x.shape

        return pool_row2output(out, B, H_out, W_out)
        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.

        dy = pool_output2row(dy)
        dx = np.zeros(self.x_shape)
        dx[range(self.x_shape[0]), self.arg_out] = dy

        return row2pool_indices(dx, self.input_shape, field_height=self.kernel_size, field_width=self.kernel_size,
                                stride=self.stride)

        # End of todo


class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.

        self.p = p
        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.

        self.mask = (np.random.ranf(x.shape) < self.p) / self.p
        return np.multiply(x, self.mask)
        # End of todo

    def backard(self, dy):

        # TODO Implement backward propogation
        # of dropout module.

        return np.multiply(dy, self.mask)
        # End of todo


if __name__ == '__main__':
    import pdb; pdb.set_trace()
