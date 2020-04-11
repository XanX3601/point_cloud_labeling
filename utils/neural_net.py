import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        """Net constructor.

        Cubes are considered of size 20 x 20 x 20 and
        as described in the paper, we use 20 filters.
        """
        super(Net, self).__init__()

        # Output classes
        self.o = 14

        # Number of channels (3 for RGB)
        self.c = 1

        # Input convolution
        self.conv_0 = nn.Conv3d(self.c, 20, (5, 5, 5), 1)

        # Max-pooling layer
        self.max_pool_0 = nn.MaxPool3d(2, 2)

        # Second convolution
        self.conv_1 = nn.Conv3d(20, 20, (5, 5, 5), 1)

        # Second max-pooling layer
        self.max_pool_1 = nn.MaxPool3d(2, 2)

        # Fully-connected layer
        self.dense_out = nn.Linear(20 * (2 * 2 * 2), self.o)

    def forward(self, x):
        """Forward function.

        In commentary, shape evolution with X the size of the batch.
        """
        # Shape (X, 1, 20, 20, 20)
        x = self.conv_0(x)

        # Shape (X, 20, 16, 16, 16)
        x = self.max_pool_0(x)
        x = torch.tanh(x)

        # Shape (X, 20, 8, 8, 8)
        x = self.conv_1(x)

        # Shape (X, 20, 4, 4, 4)
        x = self.max_pool_1(x)
        x = torch.tanh(x)

        # Shape (X, 20, 2, 2, 2)
        x = torch.flatten(x, 1, -1)
        x = self.dense_out(x)

        return nn.Softmax(dim=1)(x)
