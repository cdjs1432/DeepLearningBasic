import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class ConvLayer:
    def __init__(self, filters, kernel_size, stride=1, pad=0, initializer='he', reg=0):
        self.activation = False
        self.reg = reg
        self.x = None
        self.param = None
        self.grad = None
        self.stride = stride
        self.pad = pad
        self.init = initializer
        self.kernel_size = kernel_size
        self.filters = filters
        self.col = None
        self.col_param = None
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.out = (filters, *kernel_size)

    def forward(self, x):
        # Convolution Calculation
        self.x = x
        fn, fc, fh, fw = self.param.shape
        n, c, h, w = x.shape

        out_h = int(1 + (h + 2 * self.pad - fh) / self.stride)
        out_w = int(1 + (w + 2 * self.pad - fw) / self.stride)

        # Conv Input Size: (Channel, Filter_num, kernel_h, kernel_w)
        # Change this using im2col
        col = im2col(x, self.kernel_size[0], self.kernel_size[1], self.stride, self.pad)
        col_param = self.param.reshape((fn, -1)).T

        self.col = col
        self.col_param = col_param
        out = np.dot(col, col_param)
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        fn, c, fh, fw = self.param.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, fn)

        self.grad = np.dot(self.col.T, dout)
        self.grad = self.grad.transpose(1, 0).reshape(fn, c, fh, fw)

        dcol = np.dot(dout, self.col_param.T)
        dx = col2im(dcol, self.x.shape, fh, fw, self.stride, self.pad)

        return dx
