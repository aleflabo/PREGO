import numpy as np
import torch
import random
import math

from itertools import product
from torch import log, exp
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_all_costs(
    sample,
    distractor,
    gamma_xz,
    drop_cost_type,
    keep_percentile,
    l2_normalize=False,
    given_baseline_logits=None,
    return_baseline=False,
    do_logsoftmax=True,
    object="step",
):
    """This function computes pairwise match and individual drop costs used in Drop-DTW

    Parameters
    __________

    sample: dict
        sample dictionary
    distractor: torch.tensor of size [d] or None
        Background class prototype. Only used if the drop cost is learnable.
    distractor: torch.tensor of size [d] or None
        Background class prototype. Only used if the drop cost is learnable.
    drop_cost_type: str
        The type of drop cost definition, i.g., learnable or logits percentile.
    keep_percentile: float in [0, 1]
        if drop_cost_type == 'logit', defines drop (keep) cost threshold as logits percentile
    l2_normalize: bool
        wheather to normalize clip and step features before computing the costs
    """

    labels = sample[f"{object}_ids"]
    step_features, frame_features = sample[f"{object}_features"], sample["frame_features"]
    if l2_normalize:
        frame_features = F.normalize(frame_features, p=2, dim=1)
        step_features = F.normalize(step_features, p=2, dim=1)
    sim = step_features @ frame_features.T

    unique_labels, unique_index, unique_inverse_index = np.unique(
        labels.detach().cpu().numpy(), return_index=True, return_inverse=True
    )
    unique_sim = sim[unique_index]

    if drop_cost_type == "logit":
        k = max([1, int(torch.numel(unique_sim) * keep_percentile)])
        if given_baseline_logits is not None:
            baseline_logits = given_baseline_logits
        else:
            baseline_logit = torch.topk(unique_sim.reshape([-1]), k).values[-1].detach()
            baseline_logits = baseline_logit.repeat([1, unique_sim.shape[1]])  # making it of shape [1, N]
        sims_ext = torch.cat([unique_sim, baseline_logits], dim=0)
    elif drop_cost_type == "learn":
        distractor_sim = frame_features @ distractor
        sims_ext = torch.cat([unique_sim, distractor_sim[None, :]], dim=0)
    else:
        assert False, f"No such drop mode {drop_cost_type}"

    if do_logsoftmax:
        unique_softmax_sims = torch.nn.functional.softmax(sims_ext / gamma_xz, dim=0)
        unique_softmax_sim, drop_probs = unique_softmax_sims[:-1], unique_softmax_sims[-1]
        matching_probs = unique_softmax_sim[unique_inverse_index]
        zx_costs = -torch.log(matching_probs + 1e-5)
        drop_costs = -torch.log(drop_probs + 1e-5)
    else:
        zx_costs = -sim
        drop_costs = -baseline_logits[0]
        drop_probs = None

    if return_baseline:
        return zx_costs, drop_costs, drop_probs, baseline_logits
    else:
        return zx_costs, drop_costs, drop_probs


class VarTable:
    def __init__(self, dims, dtype=torch.float, device=device):
        self.dims = dims
        d1, d2, d_rest = dims[0], dims[1], dims[2:]

        self.vars = []
        for i in range(d1):
            self.vars.append([])
            for j in range(d2):
                var = torch.zeros(d_rest).to(dtype).to(device)
                self.vars[i].append(var)

    def __getitem__(self, pos):
        i, j = pos
        return self.vars[i][j]

    def __setitem__(self, pos, new_val):
        i, j = pos
        if self.vars[i][j].sum() != 0:
            assert False, "This cell has already been assigned. There must be a bug somwhere."
        else:
            self.vars[i][j] = self.vars[i][j] + new_val

    def show(self):
        device, dtype = self[0, 0].device, self[0, 0].dtype
        mat = torch.zeros((self.d1, self.d2, self.d3)).to().to(dtype).to(device)
        for dims in product([range(d) for d in self.dims]):
            i, j, rest = dims[0], dims[1], dims[2:]
            mat[dims] = self[i, j][rest]
        return mat


def minGamma(inputs, gamma=1, keepdim=True):
    """continuous relaxation of min defined in the D3TW paper"""
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minG = inputs.min(dim=0, keepdim=keepdim)
    else:
        # log-sum-exp stabilization trick
        zi = -inputs / gamma
        max_zi = zi.max()
        log_sum_G = max_zi + log(exp(zi - max_zi).sum(dim=0, keepdim=keepdim) + 1e-5)
        minG = -gamma * log_sum_G
    return minG


def minProb(inputs, gamma=1, keepdim=True):
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minP = inputs.min(dim=0, keepdim=keepdim)
    else:
        probs = F.softmax(-inputs / gamma, dim=0)
        minP = (probs * inputs).sum(dim=0, keepdim=keepdim)
    return minP


def traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)
