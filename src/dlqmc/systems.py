import chemcoord
import numpy as np
import pandas as pd

from deepqmc.molecule import Molecule, angstrom


def from_zmat(zmat, **kwargs):
    zmat = chemcoord.Zmat(
        pd.DataFrame(zmat, columns='atom b bond a angle d dihedral'.split())
    )
    cart = zmat.get_cartesian()
    coords = cart[['x', 'y', 'z']].values.astype(np.float32) * angstrom
    charges = cart['atom']
    return Molecule(coords, charges, **kwargs)


def cyclobutadiene_ground():
    rcc2, rcc1, rch, acch = 1.354, 1.564, 1.079, 134.94
    zmat = [
        [6, 'origin', 0.0, 'e_x', 0.0, 'e_y', 0.0],
        [6, 0, rcc1, 'e_x', 0.0, 'e_y', 0.0],
        [6, 1, rcc2, 0, 90.0, 'e_y', 0.0],
        [6, 2, rcc1, 1, 90.0, 'e_y', 0.0],
        [1, 0, rch, 1, acch, 2, 180.0],
        [1, 1, rch, 0, acch, 3, 180.0],
        [1, 2, rch, 3, acch, 0, 180.0],
        [1, 3, rch, 2, acch, 1, 180.0],
    ]
    return from_zmat(zmat, charge=0, spin=0)


def cyclobutadiene_transition():
    rcc, rch = 1.451, 1.078
    zmat = [
        [6, 'origin', 0.0, 'e_x', 0.0, 'e_y', 0.0],
        [6, 0, rcc, 'e_x', 0.0, 'e_y', 0.0],
        [6, 1, rcc, 0, 90.0, 'e_y', 0.0],
        [6, 2, rcc, 1, 90.0, 'e_y', 0.0],
        [1, 0, rch, 1, 135.0, 2, 180.0],
        [1, 1, rch, 0, 135.0, 3, 180.0],
        [1, 2, rch, 3, 135.0, 0, 180.0],
        [1, 3, rch, 2, 135.0, 1, 180.0],
    ]
    return from_zmat(zmat, charge=0, spin=0)
