import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T
from torch.nn.parameter import Parameter


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, num_players = 3):
        super(DQN, self).__init__()
        self.num_players = num_players
        # Define basic fully connected network for parameters in Advantage function
        self.main = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 60),
            nn.ReLU(),
            nn.Linear(60, 160),
            nn.ReLU(),
            nn.Linear(160, 60),
            nn.ReLU(),
            nn.Linear(60, output_dim)
        )

        # Define basic fully connected network for estimating nash value of each state
        self.main_V = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, self.num_players)
        )

        self.main_VMinus = nn.Sequential(
            nn.Linear(input_dim + self.num_players - 1, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )


class BlockPermInvariantLinear(torch.nn.Module):
    """
    This  an nn.Linear layer that is invariant to block permutations of the input.
    Input is `in_features`-Dimensional and Output is `out_features`-dimensional.
    Assume input is composed of blocks [A1,A2,...,An] each of size `block_size`.
    By default, `block_size` is assumed to be 1..
    """

    def __init__(self, in_features, out_features, block_size=1, bias=True):

        super(BlockPermInvariantLinear, self).__init__()

        # Store input and output dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        assert (self.in_features) % (self.block_size), "in_features must be a multiple of block size."

        # Generate one row of rank-1 matrix, store as parameter, as well as bias
        # Register both as parameters
        self.weight = Parameter(torch.Tensor(out_features, block_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Define full matrix used in linear transformation, uses repeated single row
        self.rep_weight = self.weight.repeat(1, self.in_features / self.block_size)

    def forward(self, x):
        return F.linear(input, self.rep_weight, self.bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class PermInvariantQNN(torch.nn.Module):
    def __init__(self, in_invar_dim, non_invar_dim, out_dim, block_size=1, num_moments=5):
        super(PermInvariantQNN, self).__init__()

        # Store input and output dimensions
        self.in_invar_dim = in_invar_dim
        self.non_invar_dim = non_invar_dim
        self.block_size = block_size
        self.num_moments = num_moments
        self.out_dim = out_dim

        # Verify invariant dimension is multiple of block size
        assert (self.in_invar_dim) % (self.block_size), "in_invar_dim must be a multiple of block size."

        # Compute Number of blocks
        self.num_blocks = self.in_invar_dim / self.block_size

        # Define Networks
        self.moment_encoder_net = nn.Sequential(
            nn.Linear(self.block_size, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.num_moments)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.num_moments + self.non_invar_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.out_dim)
        )

    def forward(self, invar_input, non_invar_input):
        # Reshape invar_input into blocks and compute "moments"
        invar_split = torch.split(invar_input, self.block_size, dim=-1)
        invar_moments = sum((self.moment_encoder_net(ch) for ch in invar_split))
        invar_moments = invar_moments / self.num_blocks

        # Concat moment vector with non-invariant input and pipe into next layer
        cat_input = torch.cat((invar_moments, non_invar_input), dim=-1)

        # Output Final Tensor
        out_tensor = self.decoder_net(cat_input)

        return out_tensor
