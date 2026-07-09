
from __future__ import annotations
import numpy as np
import os
import math
import logging
import random
import datetime
import socket

import torch
import torch.nn as nn
from torch.nn import functional as F
# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
import math
from abc import ABC
from dataclasses import dataclass
from typing import Sequence, Tuple


from torch import nn
torch.pi = torch.acos(torch.zeros(1)).item() * 2



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def new_log(logdir, filename):
    """Defines logging format.
    """
    filename = os.path.join(logdir,
                            datetime.datetime.now().strftime(
                                "log_%Y-%m-%d-%H-%M-%S_" + socket.gethostname() + "_" + filename + ".log"))
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        format="%(asctime)s - %(name)s - %(message)s",
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)



def haversine(input_coords,
              pred_coords):
    """ Calculate the haversine distances between input_coords and pred_coords.

    Args:
        input_coords, pred_coords: Tensors of size (...,N), with (...,0) and (...,1) are
        the latitude and longitude in radians.

    Returns:
        The havesine distances between
    """
    R = 6371
    lat_errors = pred_coords[..., 0] - input_coords[..., 0]
    lon_errors = pred_coords[..., 1] - input_coords[..., 1]
    a = torch.sin(lat_errors / 2) ** 2 \
        + torch.cos(input_coords[:, :, 0]) * torch.cos(pred_coords[:, :, 0]) * torch.sin(lon_errors / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    d = R * c
    return d




def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """Keep only k values nearest the current idx.
    
    Args:
        att_logits: a Tensor of shape (bachsize, data_size).
        att_idxs: a Tensor of shape (bachsize, 1), indicates
            the current idxs.
        r_vicinity: number of values to be kept.
    """
    device = att_logits.device
    idx_range = torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0], 1)
    idx_dists = torch.abs(idx_range - att_idxs)
    out = att_logits.clone()
    out[idx_dists >= r_vicinity / 2] = -float('Inf')

    return out
class UpProjConfigMixin:
    proj_factor: float = None  # will be overridden by subclasses
    round_proj_up_dim_up: bool = True
    round_proj_up_to_multiple_of: int = 64

    # internal
    _proj_up_dim: int = None  # will be computed from embedding_dim and proj_factor

    def _set_proj_up_dim(self, embedding_dim: int) -> None:
        if self.proj_factor is not None and embedding_dim is not None:
            proj_up_dim = self.proj_factor * embedding_dim
            multiple_of_multiplier = proj_up_dim / self.round_proj_up_to_multiple_of
            if self.round_proj_up_dim_up:
                multiple_of_multiplier = math.ceil(multiple_of_multiplier)
            else:
                multiple_of_multiplier = math.floor(multiple_of_multiplier)

            self._proj_up_dim = int(multiple_of_multiplier * self.round_proj_up_to_multiple_of)
class WeightDecayOptimGroupMixin(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_weight_decay_optim_groups(self, **kwargs) -> Tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        """Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight decay.
        Performs checks to ensure that each parameter is only in one of the two sequences.
        """
        weight_decay, no_weight_decay = self._create_weight_decay_optim_groups(**kwargs)

        # Check that parameters have been assigned correctly.
        # Each parameter can only be in one optim group.
        intersection_params = set(weight_decay).intersection(set(no_weight_decay))
        assert (
            len(intersection_params) == 0
        ), f"parameters {[pn for pn, p in self.named_parameters() if p in intersection_params]} made it into both decay/no_decay sets!"

        union_params = set(weight_decay).union(set(no_weight_decay))
        param_dict = {pn: p for pn, p in self.named_parameters()}
        unassigned_params = set(param_dict.values()) - union_params
        unassigned_params = [up for up in unassigned_params if not hasattr(up, "requires_grad") or up.requires_grad]
        # We have parameters that were not assigned to either weight decay or no weight decay.
        # Find the parameter names and raise an error.
        assert (
            len(unassigned_params) == 0
        ), f"Parameters {[pn for pn, p in self.named_parameters() if all([p is not q for q in unassigned_params])]} were not separated into either decay/no_decay set!"

        return weight_decay, no_weight_decay

    def get_weight_decay_optim_group_param_names(self, **kwargs) -> Tuple[Sequence[str], Sequence[str]]:
        """Return a tuple of two sequences, one for parameter names with weight decay and one for parameter names without weight decay.
        Performs checks to ensure that each parameter is only in one of the two sequences.
        """

        def _is_in_sequence(param: nn.Parameter, sequence: Sequence[nn.Parameter]) -> bool:
            for p in sequence:
                if param is p:
                    return True
            return False

        weight_decay, no_weight_decay = self.get_weight_decay_optim_groups(**kwargs)
        names_weight_decay = [pn for pn, p in self.named_parameters() if _is_in_sequence(p, weight_decay)]
        names_no_weight_decay = [pn for pn, p in self.named_parameters() if _is_in_sequence(p, no_weight_decay)]
        return names_weight_decay, names_no_weight_decay

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        """Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight decay.
        Default separation:
        - weight decay: all parameters which have > 1 dimensions.
        - no weight decay: all parameters which have = 1 dimension, e.g. biases.
        """

        decay = set()
        no_decay = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                # TODO: this is a hack based on module naming. It fixes the issue with the previous heuristic using
                #  parameter dim, which fails for FSDP because it flattens parameters so ndim is always 1.
                if 'weight' in name and 'norm' not in name:
                    decay.add(param)
                else:
                    no_decay.add(param)

        return tuple(decay), tuple(no_decay)

    def _get_weight_decay_optim_groups_for_modules(
        self, modules: list["WeightDecayOptimGroupMixin"], **kwargs
    ) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = (), ()
        for module in modules:
            wd, nwd = module.get_weight_decay_optim_groups(**kwargs)
            weight_decay += wd
            no_weight_decay += nwd
        return weight_decay, no_weight_decay
