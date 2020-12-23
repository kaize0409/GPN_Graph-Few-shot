import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution

class GPN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x


class GPN_Valuator(nn.Module):
    """
    For the sake of model efficiency, the current implementation is a little bit different from the original paper.
    Note that you can still try different architectures for building the valuator network.

    """
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Valuator, self).__init__()
        
        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc3(x)

        return x
