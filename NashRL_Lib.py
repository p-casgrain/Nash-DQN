import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
from torch.nn.parameter import Parameter


class BlockPermInvariantLinear(torch.nn.Module):
    """
    This  an nn.Linear layer that is invariant to block permutations of the input.
    Input is `in_features`-Dimensional and Output is `out_features`-dimensional.
    Assume input is composed of blocks [A1,A2,...,An] each of size `block_size`.
    By default, `block_size` is assumed to be 1..
    """
    def __init__(self, in_features, out_features, block_size = 1, bias=True):

        super(PermInvariantLinear, self).__init__()

        # Store input and output dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        assert  (self.in_features)%(self.block_size), "in_features must be a multiple of block size."

        # Generate one row of rank-1 matrix, store as parameter, as well as bias
        # Register both as parameters
        self.weight = Parameter(torch.Tensor(out_features, block_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Define full matrix used in linear transformation, uses repeated single row
        self.rep_weight = self.weight.repeat(1,self.in_features/self.block_size)

    def forward(self, x):
        return F.linear(input, self.rep_weight, self.bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)