""" 
Implementation of MLP layer of DA360

"""

import torch 
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dims):
        """Initialize MLP

        Args:
            input_dims (np.array): shape of input layer. For DA360, it will be n x num_classes, 
            where n is ___ and num_classes is the input class token dimension    
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, input_dims//2),
            nn.ReLU(),
            nn.Linear(input_dims//2, input_dims//4),
            nn.ReLU(),
            nn.Linear(input_dims//4, 1)
        )
        return

    def forward(self, input):
        return self.mlp(input)