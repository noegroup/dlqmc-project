import inspect

import tomlkit

from deepqmc import evaluate, train
from deepqmc.fit import fit_wf
from deepqmc.sampling import LangevinSampler, sample_wf
from deepqmc.utils import NULL_DEBUG
from deepqmc.wf import PauliNet
from deepqmc.wf.paulinet import ElectronicSchNet, OmniSchNet, SubnetFactory

DEEPQMC_MAPPING = {
    (train, 'sampler_kwargs'): LangevinSampler.from_mf,
    (train, 'fit_kwargs'): fit_wf,
    (LangevinSampler.from_mf, 'kwargs'): LangevinSampler,
    (evaluate, 'sampler_kwargs'): (
        LangevinSampler.from_mf,
        ['n_decorrelate', 'n_discard', 'sample_size'],
    ),
    (evaluate, 'sample_kwargs'): sample_wf,
    (PauliNet.from_hf, 'pauli_kwargs'): (
        PauliNet.from_pyscf,
        ['cusp_correction', 'cusp_electrons'],
    ),
    (PauliNet.from_hf, 'omni_kwargs'): OmniSchNet,
    (PauliNet.from_pyscf, 'kwargs'): PauliNet,
    (OmniSchNet, 'schnet_kwargs'): ElectronicSchNet,
    (OmniSchNet, 'subnet_kwargs'): SubnetFactory,
}


def _get_subkwargs(func, name, mapping):
    target = mapping[func, name]
    target, override = target if isinstance(target, tuple) else (target, [])
    sub_kwargs = collect_kwarg_defaults(target, mapping)
    for k in override:
        del sub_kwargs[k]
    return sub_kwargs


def collect_kwarg_defaults(func, mapping):
    kwargs = tomlkit.table()
    for p in inspect.signature(func).parameters.values():
        if p.name == 'kwargs':
            assert p.default is p.empty
            assert p.kind is inspect.Parameter.VAR_KEYWORD
            sub_kwargs = _get_subkwargs(func, 'kwargs', mapping)
            for item in sub_kwargs.value.body:
                kwargs.add(*item)
        elif p.name.endswith('_kwargs'):
            assert p.default is None
            assert p.kind is inspect.Parameter.KEYWORD_ONLY
            sub_kwargs = _get_subkwargs(func, p.name, mapping)
            kwargs[p.name] = sub_kwargs
        elif p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            assert p.default in (p.empty, p.default, NULL_DEBUG)
        else:
            assert p.kind is inspect.Parameter.KEYWORD_ONLY
            if p.default is None:
                kwargs.add(tomlkit.comment(f'{p.name} = ...'))
            else:
                if isinstance(p.default, tuple):
                    default = list(p.default)
                else:
                    default = p.default
                try:
                    kwargs[p.name] = default
                except ValueError:
                    print(func, p.name, p.kind, p.default)
                    raise
    return kwargs
