from pathlib import Path

import toml

from deepqmc import Molecule
from deepqmc.wf import PauliNet


def wf_from_file(path, state=None):
    param = toml.loads(Path(path).read_text())
    assert not (param.keys() - {'system', 'pauli_kwargs', 'train_kwargs'})
    system = param['system']
    if isinstance(system, str):
        system = {'name': system}
    mol = Molecule.from_name(**system)
    wf = PauliNet.from_hf(mol, **param.get('pauli_kwargs', {}))
    if state:
        wf.load_state_dict(state['wf'])
    return wf, param.get('train_kwargs', {})
