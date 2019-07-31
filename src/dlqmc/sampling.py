import numpy as np
import pandas as pd
import torch

from . import torchext
from .physics import clean_force, quantum_force
from .utils import assign_where


def samples_from(sampler, steps, *, n_discard=0, n_decorrelate=0):
    rs, psis, infos = zip(
        *(
            samples
            for i, samples in zip(steps, sampler)
            if i >= n_discard and i % (n_decorrelate + 1) == 0
        )
    )
    return torch.stack(rs, dim=1), torch.stack(psis, dim=1), pd.DataFrame(infos)


class LangevinSampler:
    def __init__(
        self, wf, rs, *, tau, max_age=None, n_first_certain=0, psi_threshold=None
    ):
        self.wf, self.rs, self.tau = wf, rs.clone(), tau
        self.max_age = max_age
        self.n_first_certain = n_first_certain
        self.psi_threshold = psi_threshold
        self.restart()

    def __len__(self):
        return len(self.rs)

    def __iter__(self):
        return self

    def _walker_step(self):
        return (
            self.rs
            + self.forces * self.tau
            + torch.randn_like(self.rs) * np.sqrt(self.tau)
        )

    def qforce(self, rs):
        try:
            forces, psis = quantum_force(rs, self.wf)
        except torchext.LUFactError as e:
            e.info['rs'] = rs[e.info['idxs']]
            raise
        forces = clean_force(forces, rs, self.wf.geom, tau=self.tau)
        return forces, psis

    def __next__(self):
        rs_new = self._walker_step()
        forces_new, psis_new = self.qforce(rs_new)
        log_G_ratios = (
            (self.forces + forces_new)
            * ((self.rs - rs_new) + self.tau / 2 * (self.forces - forces_new))
        ).sum(dim=(-1, -2))
        Ps_acc = torch.exp(log_G_ratios) * (psis_new / self.psis) ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        if self.psi_threshold is not None:
            accepted = accepted & (psis_new.abs() > self.psi_threshold) | (
                (self.psis.abs() < self.psi_threshold)
                & (psis_new.abs() > self.psis.abs())
            )
        if self.max_age is not None:
            accepted = accepted | (self._ages >= self.max_age)
        if self._step < self.n_first_certain:
            accepted = torch.ones_like(accepted)
        self._ages[accepted] = 0
        self._ages[~accepted] += 1
        info = {
            'acceptance': accepted.type(torch.int).sum().item() / self.rs.shape[0],
            'age': self._ages.cpu().numpy(),
        }
        assign_where(
            (self.rs, self.psis, self.forces), (rs_new, psis_new, forces_new), accepted
        )
        self._step += 1
        return self.rs.clone(), self.psis.clone(), info

    def __repr__(self):
        return (
            f'<LangevinSampler n_walker={self.rs.shape[0]} '
            'n_electrons={self.rs.shape[1]} tau={self.tau}>'
        )

    def propagate_all(self):
        self.rs = self._walker_step()
        self.restart()

    def restart(self):
        self._step = 0
        self.forces, self.psis = self.qforce(self.rs)
        self._ages = torch.zeros_like(self.psis, dtype=torch.long)


def rand_from_mf(mf, bs, charge_std=0.25, elec_std=1.0, idxs=None):
    mol = mf.mol
    n_atoms = mol.natm
    charges = mol.atom_charges()
    n_electrons = charges.sum() - mol.charge
    while idxs is None:
        cs = torch.tensor(charges - mf.pop(verbose=0)[1]).float()
        cs = cs + charge_std * torch.randn(bs, n_atoms)
        repeats = (cs / cs.sum(dim=-1)[:, None] * n_electrons).round().to(torch.long)
        try:
            idxs = torch.repeat_interleave(
                torch.arange(n_atoms).expand(bs, -1), repeats.flatten()
            ).view(bs, n_electrons)
        except RuntimeError:
            continue
    idxs = torch.stack([idxs[i, torch.randperm(idxs.shape[-1])] for i in range(bs)])
    centers = torch.tensor(mol.atom_coords()).float()[idxs]
    rs = centers + elec_std * torch.randn_like(centers)
    return rs
