import torch
import torch.nn as nn
from models.adaIN import AdaptiveInstanceNormalization

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding = 0,
                 norm = 'none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNormalization(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization type: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation == nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'tanh':
            self.activation == nn.Tanh()
        elif activation == 'none':
            self.activation == None
        else:
            assert 0, "Unsupported activation type: {}".format(activation)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                              bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, norm = 'in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(input_dim=dim, output_dim=dim, kernel_size=3, stride=1, padding=1,
                              norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(input_dim=dim, output_dim=dim, kernel_size=3, stride=1, padding=1,
                              norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation ='relu', pad_type = 'zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim=dim, norm='in', activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)