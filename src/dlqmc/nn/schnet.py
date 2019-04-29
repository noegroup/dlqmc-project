import numpy as np
import torch
import torch.nn as nn

from ..utils import NULL_DEBUG, nondiag
from .base import SSP


class ZeroDiagKernel(nn.Module):
    def forward(self, Ws):
        Ws = Ws.clone()
        i, j = np.diag_indices(Ws.shape[1])
        Ws[:, i, j] = 0
        return Ws


def get_schnet_interaction(kernel_dim, embedding_dim, basis_dim):
    modules = {
        'kernel': nn.Sequential(
            nn.Linear(basis_dim, kernel_dim),
            SSP(),
            nn.Linear(kernel_dim, kernel_dim),
            ZeroDiagKernel(),
        ),
        'embed_in': nn.Linear(embedding_dim, kernel_dim, bias=False),
        'embed_out': nn.Sequential(
            nn.Linear(kernel_dim, embedding_dim),
            SSP(),
            nn.Linear(embedding_dim, embedding_dim),
        ),
    }
    return nn.ModuleDict(modules)


# TODO test if faster than ZeroDiagKernel
def schnet_conv(Ws, zs):
    i, j = np.mask_indices(Ws.shape[2], nondiag)
    n = Ws.shape[1] * (Ws.shape[2] - 1)
    i, j = i[:n], j[:n]
    return (
        Ws[:, i, j].view(*Ws.shape[:1], Ws.shape[2] - 1, -1)
        * zs[:, j].view(*Ws.shape[:1], Ws.shape[2] - 1, -1)
    ).sum(dim=2)


class ElectronicSchnet(nn.Module):
    def __init__(
        self,
        n_up,
        n_down,
        n_nuclei,
        n_interactions,
        basis_dim,
        kernel_dim,
        embedding_dim,
    ):
        super().__init__()
        self.embedding_nuc = nn.Parameter(torch.randn(n_nuclei, kernel_dim))
        self.embedding_elec = nn.Parameter(
            torch.cat(
                [torch.randn(embedding_dim).expand(n_el, -1) for n_el in (n_up, n_down)]
            )
        )
        self.interactions = nn.ModuleList(
            [
                get_schnet_interaction(kernel_dim, embedding_dim, basis_dim)
                for _ in range(n_interactions)
            ]
        )

    def forward(self, dists_basis, debug=NULL_DEBUG):
        bs = len(dists_basis)  # batch size
        xs = debug[0] = self.embedding_elec.clone().expand(bs, -1, -1)
        for i, interaction in enumerate(self.interactions):
            Ws = interaction.kernel(dists_basis)
            zs = interaction.embed_in(xs)
            zs = torch.cat([zs, self.embedding_nuc.expand(bs, -1, -1)], dim=1)
            zs = (Ws * zs[:, None, :, :]).sum(dim=2)
            xs = debug[i + 1] = xs + interaction.embed_out(zs)
        return xs
