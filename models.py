import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable

class KoopmanOperator(nn.Module):
    def __init__(self,koopman_dim,delta_t,device="cpu"):
        super(KoopmanOperator,self).__init__()

        self.koopman_dim = koopman_dim
        self.num_eigenvalues = int(koopman_dim/2)
        self.delta_t = delta_t
        self.parameterization = nn.Sequential(
            nn.Linear(self.koopman_dim,self.num_eigenvalues*2),
            nn.Tanh(),
            nn.Linear(self.num_eigenvalues*2,self.num_eigenvalues*2)
        )
        self.device = device

    def forward(self,x,T):
        # x is B  x Latent
        # it is the one because only initial point (T=1)

        # mu is B  x Latent/2
        # omega is B  x Latent/2

        Y = Variable(torch.zeros(x.shape[0],T,self.koopman_dim)).to(self.device)
        y = x[:,0,:]
        for t in range(T):
            mu,omega = torch.unbind(self.parameterization(y).reshape(-1,self.num_eigenvalues,2),-1)

            # K is B x Latent x Latent

            # B x Koopmandim/2
            exp = torch.exp(self.delta_t * mu)

            # B x Latent/2
            cos = torch.cos(self.delta_t * omega)
            sin = torch.sin(self.delta_t * omega)


            K = Variable(torch.zeros(x.shape[0],self.koopman_dim,self.koopman_dim)).to(self.device)

            for i in range(0,self.koopman_dim,2):
                #for j in range(i,i+2):
                index = i//2

                K[:, i + 0, i + 0] = cos[:,index] *  exp[:,index]
                K[:, i + 0, i + 1] = -sin[:,index] * exp[:,index]
                K[:, i + 1, i + 0] = sin[:,index]  * exp[:,index]
                K[:, i + 1, i + 1] = cos[:,index] * exp[:,index]

            y = torch.matmul(K,y.unsqueeze(-1)).squeeze(-1)

            Y[:,t,:] = y

        return Y


class Lusch(nn.Module):
    def __init__(self,input_dim,koopman_dim,hidden_dim,delta_t=0.01,device="cpu"):
        super(Lusch,self).__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim,koopman_dim))

        self.decoder = nn.Sequential(nn.Linear(koopman_dim,hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim,input_dim))

        self.koopman = KoopmanOperator(koopman_dim,delta_t,device)

        self.device = device
        self.delta_t = delta_t

        # Normalization occurs inside the model
        self.register_buffer('mu', torch.zeros((input_dim,)))
        self.register_buffer('std', torch.ones((input_dim,)))

    def forward(self,x):
        x = self.embed(x)
        x = self.recover(x)
        return x

    def embed(self,x):
        x = self._normalize(x)
        x = self.encoder(x)
        return x

    def recover(self,x):
        x = self.decoder(x)
        x = self._unnormalize(x)
        return x

    def koopman_operator(self,x,T=1):
        return self.koopman(x,T)

    def _normalize(self, x):
        return (x - self.mu[(None,)*(x.dim()-1)+(...,)])/self.std[(None,)*(x.dim()-1)+(...,)]

    def _unnormalize(self, x):
        return self.std[(None,)*(x.dim()-1)+(...,)]*x + self.mu[(None,)*(x.dim()-1)+(...,)]