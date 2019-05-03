import torch
import torch.nn as nn

class PermInvariantQNN(torch.nn.Module):
    """
    Permutation Invariant Network
    
    :param in_invar_dim:   Number of total features across all agents that needs to be perm invariant 
    :param non_invar_dim:  Number of total features constant across all agents
    :param out_dim:        Dimension of output
    :param block_size:     Number of invariant features of each agent
    :param num_moments:    Number of features/moments to summarize invariant features of each agent
    :raises assertError:   Raise assertion error if in_invar_dim not multiple of block size
    """
    
    block_size: int
    in_invar_dim: int
    non_invar_dim: int
    num_moments: int
    out_dim: int

    def __init__(self, in_invar_dim, non_invar_dim,
                 out_dim, block_size=1, num_moments=1):
        super(PermInvariantQNN, self).__init__()

        # Store input and output dimensions
        self.in_invar_dim = in_invar_dim
        self.non_invar_dim = non_invar_dim
        self.block_size = block_size
        self.num_moments = num_moments
        self.out_dim = out_dim

        # Verify invariant dimension is multiple of block size
        assert not self.in_invar_dim % self.block_size, "in_invar_dim must be a multiple of block size."

        # Compute Number of blocks
        self.num_blocks = self.in_invar_dim / self.block_size

        # Define Networks
        self.moment_encoder_net = nn.Sequential(
            nn.Linear(self.block_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.num_moments),
            #nn.BatchNorm1d(self.num_moments)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.num_moments + self.non_invar_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.out_dim)
        )

    def forward(self, invar_input, non_invar_input):
        # Reshape invar_input into blocks and compute "moments"
        invar_moments = torch.sum(invar_input,dim = 1).view(-1,1)
        invar_moments = invar_moments / self.num_blocks

        # Concat moment vector with non-invariant input and pipe into next layer
        cat_input = torch.cat((invar_moments, non_invar_input), dim=1)

        # Output Final Tensor
        out_tensor = self.decoder_net(cat_input)
        return out_tensor