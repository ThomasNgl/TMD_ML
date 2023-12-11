import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
    
#########
#Classifier
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens_dim, activation = F.relu):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hiddens_dim
        self.activation = activation
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hiddens_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            s = layer(x)
            x = self.activation(s)
        out = self.layers[-1](x)
        return out  

#########
#Model
class PerslayModel(nn.Module):
    def __init__(self, weight_function, point_transform, 
                 operator, 
                 output_dim, hiddens_dim, activation = F.relu, 
                 is_tensor = False):
        super(PerslayModel, self).__init__()

        self.weight_function = weight_function
        self.point_transform = point_transform

        self.operator = operator

        self.op_dim = self.operator.op_dim
        self.input_dim = self.point_transform.vect_dim
        
        self.output_dim = output_dim
        self.hiddens_dim = hiddens_dim
        self.activation = activation
        self.net = Net(self.op_dim * self.input_dim, 
                       output_dim, 
                       hiddens_dim, activation)
        
        self.is_tensor = is_tensor

    def compute(self, x):
        v = self.point_transform.forward(x)
        w = self.weight_function.forward(x) 
        z = w*v
        op = self.operator.forward(z)
        y = self.net.forward(op)
        return y
    
    def forward(self, x):
        if self.is_tensor:
            pred = self.compute(x)
        else:
            outputs = []
            for sample in x:
                x = torch.tensor(sample, dtype = torch.float32) 
                y = self.compute(x)
                outputs.append(y)
            pred = torch.stack(outputs)
        return pred

