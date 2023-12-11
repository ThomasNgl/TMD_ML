import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 

class CombineOP(nn.Module):
    def __init__(self, op_list):
        super(CombineOP, self).__init__()
        self.op_list = op_list
        self.op_dim = len(self.op_list)
    def forward(self, x):
        ops = []
        for op in self.op_list:
            vect = op.forward(x)
            ops.append(vect)
        ops = torch.concatenate((ops), axis = -1)
        return ops
    
class SumOP(nn.Module):
    def __init__(self):
        super(SumOP, self).__init__()
        self.op_dim = 1
    def forward(self, x):
        sum = torch.sum(x, dim=-2)
        return sum  

class QuantileOP(nn.Module):
    def __init__(self, q):
        super(QuantileOP, self).__init__()
        self.q = q
        self.op_dim = 1
    def forward(self, x):
        sum = torch.quantile(x, self.q, dim=-2)
        return sum

class MeanOP(nn.Module):
    def __init__(self):
        super(MeanOP, self).__init__()
        self.op_dim =1
    def forward(self, x):
        mean = torch.mean(x, dim=-2)
        return mean