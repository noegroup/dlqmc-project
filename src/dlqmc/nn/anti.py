import numpy as np
import torch
import torch.nn as nn


class PairConcat(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x1s, x2s):
        xs = torch.cat((x1s, x2s), dim=-1)
        return self.net(xs)


class NetPairwiseAntisymmetry(nn.Module):
    def __init__(self, net_pair):
        super().__init__()
        self.net_pair = net_pair

    def forward(self, x_i, x_j):
        return self.net_pair(x_i, x_j) - self.net_pair(x_j, x_i)


class NetOdd(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) - (self.net(-x))


class AntisymmetricPart(nn.Module):
    def __init__(self, net, net_pair):
        super().__init__()
        self.net_pair_anti = NetPairwiseAntisymmetry(net_pair)
        self.net_odd = NetOdd(net)

    def forward(self, x):
        i, j = np.triu_indices(x.shape[-2], k=1)
        zs = self.net_pair_anti(x[:, j], x[:, i]).prod(dim=-2)
        return self.net_odd(zs)
