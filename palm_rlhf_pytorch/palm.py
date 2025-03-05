from __future__ import annotations

import math
import copy
from pathlib import Path
from collections import namedtuple
from functools import wraps
from itertools import zip_longest

from tqdm import tqdm
from beartype import beartype

import torch
from torch import einsum, nn
import torch.nn.functionsal as F
from torch.nn import Module, ModuleList, ModuleDict

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from palm_rlhf_pytorch.attention import Attention
from palm_rlhf_pytorch.utils import top_p, top_k, masked_means, gumbel_sample, eval_decorator
from palm_rlhf_pytorch.lora import LoRA

# functions and decorators

def exits(val):
    return val is not None