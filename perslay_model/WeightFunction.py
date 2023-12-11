import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class WeightFunction(nn.Module):
    def __init__(self, min_bound = -2000, max_bound = 3000, grid = torch.ones((1,1)), denominator=1):
        super(WeightFunction, self).__init__()
        self.attributes = [min_bound, max_bound, grid, denominator]
        if type(grid) == int:
            self.resolution = grid
            self.w = Parameter((torch.rand(self.resolution, self.resolution)-0.5)/denominator)
        else:
            self.resolution = grid.shape[0]
            self.w = grid
        
        self.min_bound, self.max_bound = torch.tensor(min_bound), torch.tensor(max_bound)

    def index(self, birth, death, min_):
        birth_u = torch.unsqueeze(birth, -1)
        death_u = torch.unsqueeze(death, -1)
        b_index = torch.round((birth_u - min_)/(self.max_bound - self.min_bound) * (self.resolution-1))
        d_index = torch.round((death_u - min_)/(self.max_bound - self.min_bound) * (self.resolution-1))
        
        b_index = F.relu(b_index)
        d_index = F.relu(d_index)

        b_index_max = torch.ones(b_index.shape)*(self.w.shape[0]-1)
        d_index_max = torch.ones(d_index.shape)*(self.w.shape[1]-1)

        b_index = torch.minimum(b_index, b_index_max).int()
        d_index = torch.minimum(d_index, d_index_max).int()

        return b_index, d_index
    
    def forward(self, x):
        x_size = len(x.shape)
        if x_size == 3: 
            birth, death = x[:, :,0], x[:, :,1]
            min_rep = self.min_bound.repeat(birth.shape[0], birth.shape[1], 1)

        elif x_size == 2:
            birth, death = x[:,0], x[:,1]
            min_rep = self.min_bound.repeat(death.shape[0], 1)

        b_index, d_index = self.index(birth, death, min_rep)
        
        x_weight = self.w[(b_index, d_index)]
        return x_weight
