import gc
import logging
import math
import os
import re
import shutil
import time
from functools import partial
from importlib import resources
from itertools import chain, count, islice, product
from pathlib import Path

import chemcoord
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import tables
import toml
import torch
import uncertainties
from pyscf import dft, gto, mcscf, scf
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from uncertainties import ufloat
from uncertainties import unumpy as unp

import deepqmc.torchext
import dlqmc.systems
from deepqmc import Molecule, evaluate, train
from deepqmc.cli import wf_from_file
from deepqmc.fit import (
    LossEnergy,
    fit_wf,
    fit_wf_mem_test_func,
    log_clipped_outliers,
)
from deepqmc.physics import local_energy, pairwise_distance, pairwise_self_distance
from deepqmc.plugins import PLUGINS
from deepqmc.sampling import LangevinSampler, rand_from_mf, sample_wf, samples_from
from deepqmc.tools.analysis import (
    GaussianKDEstimator,
    autocorr_coeff,
    blocking,
    pair_correlations_from_samples,
)
from deepqmc.tools.memory import UNKNWON_CLASSES, find_large_cuda_tensors
from deepqmc.tools.plot import plot_func, plot_func_2d, plot_func_x, plot_func_xy
from deepqmc.torchext import normalize_mean, number_of_parameters
from deepqmc.utils import (
    DebugContainer,
    DebugLogTable,
    batch_eval,
    batch_eval_tuple,
    energy_offset,
    get_flat_mesh,
)
from deepqmc.wf import PauliNet
from deepqmc.wf.paulinet import ElectronicSchNet, OmniSchNet, SubnetFactory
from dlqmc.analysis import ewm_full, infinite_training_limit
from dlqmc.experiments import collect_all_systems
from dlqmc.plot import ewm, plot_training
from dlqmc.tools import short_fmt

logging.basicConfig(
    style='{',
    format='[{asctime}.{msecs:03.0f}] {levelname}:{name}: {message}',
    datefmt='%H:%M:%S',
)
logging.getLogger('deepqmc').setLevel(logging.DEBUG)
