# -*- coding: utf-8 -*-

# @Time    : 19-5-25 下午4:17
# @Author  : zj
import numpy as np

def get_im2row_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = stride * np.repeat(np.arange(out_height), out_width)
    i1 = np.repeat(np.arange(field_height), field_width)
    i1 = np.tile(i1, C)

    j0 = stride * np.tile(np.arange(out_width), out_height)
    j1 = np.tile(np.arange(field_width), field_height * C)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(1, -1)

    return (k, i, j)


def im2row_indices(x, field_height, field_width, padding=1, stride=1):

    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2row_indices(x.shape, field_height, field_width, padding, stride)

    rows = x_padded[:, k, i, j]
    C = x.shape[1]

    rows = rows.reshape(-1, field_height * field_width * C)
    return rows


def row2im_indices(rows, x_shape, field_height=3, field_width=3, padding=1, stride=1, isstinct=False):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=rows.dtype)
    k, i, j = get_im2row_indices(x_shape, field_height, field_width, padding,
                                 stride)
    rows_reshaped = rows.reshape(N, -1, C * field_height * field_width)
    np.add.at(x_padded, (slice(None), k, i, j), rows_reshaped)

    if isstinct:
        x_ones = np.ones(x_padded.shape)
        rows_ones = x_ones[:, k, i, j]
        x_zeros = np.zeros(x_padded.shape)
        np.add.at(x_zeros, (slice(None), k, i, j), rows_ones)
        x_padded = x_padded / x_zeros

    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]

def conv_row2output(inputs, batch_size, out_height, out_width):
    output = inputs.copy()
    # [N*H*W, C]
    local_connect_size, depth = output.shape[:2]
    # [N*H*W, C] -> [N, H, W, C]
    output = output.reshape(batch_size, out_height, out_width, depth)
    # [N, H, W, C] -> [N, C, H, W]
    return output.transpose((0, 3, 1, 2))


def conv_output2row(inputs):
    output = inputs.copy()
    # [N, C, H, W]
    num, depth, height, width = output.shape[:4]

    # [N,C,H,W] —> [N,C,H*W]
    output = output.reshape(num, depth, -1)
    # [N,C,H*W] -> [N,H*W,C]
    output = output.transpose(0, 2, 1)
    # [N,H*W,C] -> [N*H*W,C]
    return output.reshape(-1, depth)

def kernel2row(kernel):
    k, c, h, w = kernel.shape
    return kernel.reshape(k, c, 1, -1).reshape(k, 1, 1, -1).reshape(k, -1)

def row2kernel(k, kernel_shape):
    s, c, h, w = kernel_shape
    return k.T.reshape((-1, w)).reshape(-1, h, w).reshape(s, c, h, w)

def get_pool2row_indices(x_shape, field_height, field_width, stride=1):
    N, C, H, W = x_shape
    assert (H - field_height) % stride == 0
    assert (W - field_width) % stride == 0
    out_height = int((H - field_height) / stride + 1)
    out_width = int((W - field_width) / stride + 1)

    i0 = stride * np.repeat(np.arange(out_height), out_width)
    i0 = np.tile(i0, C)
    i1 = np.repeat(np.arange(field_height), field_width)

    j0 = stride * np.tile(np.arange(out_width), out_height * C)
    j1 = np.tile(np.arange(field_width), field_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), out_height * out_width).reshape(-1, 1)

    return (k, i, j)


def pool2row_indices(x, field_height, field_width, stride=1):
    k, i, j = get_pool2row_indices(x.shape, field_height, field_width, stride)
    rows = x.copy()[:, k, i, j]

    return rows.reshape(-1, field_height * field_width)


def row2pool_indices(rows, x_shape, field_height=2, field_width=2, stride=2, isstinct=False):
    N, C, H, W = x_shape
    x = np.zeros(x_shape, dtype=rows.dtype)
    k, i, j = get_pool2row_indices(x_shape, field_height, field_width, stride)
    rows_reshaped = rows.reshape(N, -1, field_height * field_width)
    np.add.at(x, (slice(None), k, i, j), rows_reshaped)

    if isstinct and (stride < field_height or stride < field_width):
        x_ones = np.ones(x.shape)
        rows_ones = x_ones[:, k, i, j]
        x_zeros = np.zeros(x.shape)
        np.add.at(x_zeros, (slice(None), k, i, j), rows_ones)
        return x / x_zeros

    return x

def pool_row2output(inputs, batch_size, out_height, out_width):
    output = inputs.copy()
    return output.reshape(batch_size, -1, out_height, out_width)

def pool_output2row(inputs):
    return inputs.copy().reshape(-1)


