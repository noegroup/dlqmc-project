from pathlib import Path

import toml

from deepqmc import Molecule
from deepqmc.wf import PauliNet


def wf_from_file(path, state=None):
    params = toml.loads(Path(path).read_text())
    system = params.pop('system')
    if isinstance(system, str):
        system = {'name': system}
    mol = Molecule.from_name(**system)
    wf = PauliNet.from_hf(mol, **params.pop('model_kwargs', {}))
    if state:
        wf.load_state_dict(state['wf'])
    return wf, params
