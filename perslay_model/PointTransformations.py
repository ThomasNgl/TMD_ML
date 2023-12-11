import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

########
#Vectorizations  
class PointTransformation(nn.Module):
    def __init__(self):
        super(PointTransformation, self).__init__()

    def forward(self, x):
        birth = x[:,:,0]
        death = x[:,:,1]
        return birth, death

class TrianglePT(nn.Module):
    def __init__(self, t = 10, denominator = 1):
    #t is a list of reals
        super(TrianglePT, self).__init__()
        self.pt = PointTransformation()
        self.attributes = [t, denominator]

        if type(t) == int:
            self.vect_dim = t
            self.t = Parameter((torch.rand(self.vect_dim)-0.5)/denominator)
        else:
            self.vect_dim = len(t)
            self.t = t

    def triangle(self, birth, death, t_rep):
        birth_u = torch.unsqueeze(birth, -1)
        death_u = torch.unsqueeze(death, -1)
        compare = death_u - abs(t_rep - birth_u)
        o = torch.zeros(compare.shape)
        x_transform = torch.max(o, compare)
        return x_transform

    def forward(self, x):
        x_size = len(x.shape)
        if x_size == 3: 
            birth, death = self.pt.forward(x)
            t_rep = self.t.repeat(birth.shape[0], birth.shape[1], 1)
        
        elif x_size == 2:
            birth, death = x[:,0], x[:,1]
            t_rep = self.t.repeat(death.shape[0], 1)
        
        x_transform = self.triangle(birth, death, t_rep)
        return x_transform

class GaussPT(nn.Module):
    def __init__(self, t = 10, sigma = None, denominator = 1):
    #t is a list of pairs   
        super(GaussPT, self).__init__()
        self.pt = PointTransformation()
        self.attributes = [t, sigma, denominator]

        if type(t) == int:
            self.vect_dim = t
            self.t = Parameter((torch.rand(self.vect_dim, 2)-0.5)/denominator) 
        else:
            self.vect_dim = len(t)**2
            self.t = torch.cartesian_prod(t, t)

        if not sigma:
            self.sigma = Parameter(torch.rand(1))
        else: 
            self.sigma = sigma

    def gaussian(self, birth, death, t_b, t_d):
        birth_u = torch.unsqueeze(birth, -1)
        death_u = torch.unsqueeze(death, -1) 
        x_transform = torch.exp( -((birth_u - t_b)**2 + (death_u - t_d)**2) / (2*self.sigma**2))
        return x_transform

    def forward(self, x):
        x_size = len(x.shape)
        if x_size == 3: 
            birth, death = self.pt.forward(x)
            t_rep = self.t.repeat(x.shape[0], x.shape[1], 1, 1)
            t_b, t_d = t_rep[:, :, :, 0], t_rep[:, :, :, 1]
        
        elif x_size == 2:
            birth, death = x[:,0], x[:,1]
            t_rep = self.t.repeat(birth.shape[0], 1, 1)
            t_b, t_d = t_rep[:, :, 0], t_rep[:, :, 1]
        
        x_transform = self.gaussian(birth, death, t_b, t_d)
        return x_transform
    
class LinePT(PointTransformation):
    def __init__(self, t = 10, denominator = 1):
    #w is a list of pair, b a list of doubles
        super(LinePT, self).__init__()     
        self.attributes = [t, denominator]
        self.t = t
        if type(t) == int:
            self.vect_dim = t
            self.w = nn.Linear(2, self.vect_dim)

        else:
            self.vect_dim = len(t)**3
            params = torch.cartesian_prod(t, t, t)
            self.w = params

    def line(self, x, w):
        x_u = x.unsqueeze(-2)
        ones = torch.ones(x_u.shape[:-1]).unsqueeze(-1)
        z = torch.concatenate((x_u,ones),axis=-1)
        x_transform = (z * w).sum(dim = -1)
        return x_transform
    
    def forward(self, x):
        x_size = len(x.shape)
        if type(self.t) == int:
            x_transform = self.w(x) 

        else:
            if x_size == 3: 
                w_rep = self.w.repeat(x.shape[0], x.shape[1], 1, 1)
            
            if x_size == 2:
                w_rep = self.w.repeat(x.shape[0], 1, 1)
            
            x_transform = self.line(x, w_rep) 

        return x_transform

class NetPT(nn.Module):
    def __init__(self, output_dim, hiddens_dim, activation = F.relu):
        super(NetPT, self).__init__()
        self.attributes = [output_dim, hiddens_dim, activation]

        self.input_dim = 2
        self.vect_dim = output_dim
        self.hidden_dim = hiddens_dim
        self.activation = activation
        current_dim = self.input_dim
        self.layers = nn.ModuleList()
        for hdim in hiddens_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def net(self, x):
        for layer in self.layers[:-1]:
            s = layer(x)
            x = self.activation(s)
        x_transform = self.layers[-1](x)
        return x_transform
    
    def forward(self, x):
        x_transform = self.net(x)
        return x_transform   
