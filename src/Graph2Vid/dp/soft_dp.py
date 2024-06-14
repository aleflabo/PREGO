import numpy as np
import torch
import math
import torch.nn.functional as F
from copy import copy

from models.model_utils import unique_softmax, cosine_sim
from dp.dp_utils import VarTable, minGamma, minProb


device = "cuda" if torch.cuda.is_available() else "cpu"


def softDTW(
    step_features,
    frame_features,
    labels,
    dist_type="inner",
    softning="prob",
    gamma_min=0.1,
    gamma_xz=0.1,
    step_normalize=True,
):
    """function to obtain a soft (differentiable) version of DTW
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    # defining the function
    _min_fn = minProb if softning == "prob" else minGamma
    min_fn = lambda x: _min_fn(x, gamma=gamma_min)

    # first get a pairwise distance matrix
    if dist_type == "inner":
        dist = step_features @ frame_features.T
    else:
        dist = cosine_sim(step_features, frame_features)
    if step_normalize:
        if labels is not None:
            norm_dist = unique_softmax(dist, labels, gamma_xz)
        else:
            norm_dist = torch.softmax(dist / gamma_xz, 0)
        dist = -log(norm_dist)

    # initialize soft-DTW table
    nrows, ncols = dist.shape
    # sdtw = torch.zeros((nrows+1,ncols+1)).to(torch.float).to(device)
    sdtw = VarTable((nrows + 1, ncols + 1))
    for i in range(1, nrows + 1):
        sdtw[i, 0] = 9999999999
    for j in range(1, ncols + 1):
        sdtw[0, j] = 9999999999

    # obtain dtw table using min_gamma or softMin relaxation
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            neighbors = torch.stack([sdtw[i, j - 1], sdtw[i - 1, j - 1], sdtw[i - 1, j]])
            di, dj = i - 1, j - 1  # in the distance matrix indices are shifted by one
            new_val = dist[di, dj] + min_fn(neighbors)
            sdtw[i, j] = torch.squeeze(new_val, 0)
    sdtw_loss = sdtw[nrows, ncols] / step_features.shape[0]
    return sdtw_loss, sdtw, dist


def dropDTW(
    step_features, frame_features, labels, softning="prob", gamma_min=1, gamma_xz=1, step_normalize=True, eps=1e-5
):
    """function to obtain a soft (differentiable version of DTW)
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    # defining the function
    _min_fn = minProb if softning == "prob" else minGamma
    min_fn = lambda x: _min_fn(x, gamma=gamma_min)

    # first get a pairwise distance matrix and drop costs
    dist = step_features @ frame_features.T
    inf = torch.tensor([9999999999], device=dist.device, dtype=dist.dtype)
    if step_normalize:
        norm_dist = unique_softmax(dist, labels, gamma_xz)
        drop_costs = 1 - norm_dist.max(dim=0).values  # assuming dist is in [0, 1]
        zx_costs = -log(norm_dist)
        drop_costs = -log(drop_costs + eps)
    else:
        zx_costs = 1 - dist
        drop_costs = 1 - zx_costs.max(dim=0).values  # assuming dist is in [0, 1]

    cum_drop_costs = torch.cumsum(drop_costs, dim=0)

    # initialize soft-DTW table
    K, N = dist.shape

    D = VarTable((K + 1, N + 1, 3))  # 3-dim DP table, instead of 3 1-dim tables above
    for i in range(1, K + 1):
        D[i, 0] = torch.zeros_like(D[i, 0]) + inf
    for j in range(1, N + 1):
        filling = torch.zeros_like(D[0, j]) + inf
        filling[[0, 1]] = cum_drop_costs[j - 1]
        D[0, j] = filling

    # obtain dtw table using min_gamma or softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            d_diag, d_left = D[zi - 1, xi - 1][0:1], D[zi, xi - 1][0:1]
            min_prev_cost = torch.zeros_like(d_diag) + min_fn([d_diag, d_left])

            # positive transition, i.e. matching x_i to z_j
            Dp = min_prev_cost + zx_costs[z_cost_ind, x_cost_ind]
            # negative transition, i.e. dropping xi
            Dm = d_left + drop_costs[x_cost_ind]

            # update final solution matrix
            D_final = torch.zeros_like(Dm) + min_fn([Dm, Dp])
            D[zi, xi] = torch.cat([D_final, Dm, Dp], dim=0)
    min_cost = D[K, N][0]

    return min_cost, D, dist


def metadag2vid_soft(zx_costs, drop_costs, metadag, idx2node, softning="prob", contiguous=True, gamma_min=1):
    """a soft (differentiable) version of graph2vid

    Parameters
    ----------
    zx_costs: np.ndarray [K, N]
        pairwise match costs between K steps and N video clips
    drop_costs: np.ndarray [N]
        drop costs for each clip
    metadag: networkx
        For each node, specifies a list of parents in the DAG.
        Assuming that the list is topologically sorted.
    exclusive: bool
        If True any clip can be matched with only one step, not many.
    contiguous: bool
        if True, can only match a contiguous sequence of clips to a step
        (i.e. no drops in between the clips)
    return_label: bool
        if True, returns output directly useful for segmentation computation (made for convenience)
    """

    # defining the min function
    min_fn = minProb if softning == "prob" else minGamma
    inf = 9999999999

    K, N = zx_costs.shape

    # Taking care of the DAG
    # prepare DAG parents in the usable format
    node2idx = {node_id: idx for idx, node_id in idx2node.items()}
    metadag_idx = dict()
    for idx, node in idx2node.items():
        parents_nodes = list(metadag.pred[node])
        parents_idxs = [node2idx[n] for n in parents_nodes]
        metadag_idx[idx] = parents_idxs

    # prepare the list of possible states to transition from
    prev_states_dict = dict()
    for node, parents in metadag_idx.items():
        if len(parents) == 0:
            prev_states_dict[node + 1] = [0]
        else:
            prev_states_dict[node + 1] = [s + 1 for s in parents]

    # Taking care of DP table initialization
    if K >= N:
        # in case the number of steps is greater than the number of frames,
        # duplicate every frame and let the drops do the job.
        mult = math.ceil(K / N)
        zx_costs = torch.stack([zx_costs] * mult, dim=-1).reshape([K, -1])
        drop_costs = torch.stack([drop_costs] * mult, dim=-1).reshape([-1])
        N *= mult
    cum_drop_costs = torch.cumsum(drop_costs, dim=0)

    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, 2), device=zx_costs.device)
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + inf
    for xi in range(1, N + 1):
        D[0, xi] = torch.zeros_like(D[0, xi]) + cum_drop_costs[(xi - 1) : xi]

    # obtain dtw table using the softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # difining costs at differenc cells
            prev_costs = []
            for pz in prev_states_dict[zi]:
                min_val = min_fn([D[pz, xi - 1][[0]], D[pz, xi - 1][[1]]], gamma=gamma_min)
                prev_costs.append(min_val)

            cur_costs = [D[zi, xi - 1][[s]] for s in [0, 1]]
            cur_pos_costs = cur_costs[:1] if contiguous else cur_costs

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            all_pos_costs = prev_costs + cur_pos_costs
            Dp = min_fn(all_pos_costs, gamma=gamma_min) + zx_costs[z_cost_ind, x_cost_ind]

            # negative transition, i.e. dropping xi
            Dn = min_fn(cur_costs, gamma=gamma_min) + drop_costs[x_cost_ind]

            # update final solution matrix
            D[zi, xi] = torch.cat([Dp, Dn], dim=0)

    # Computing the final min cost for the whole batch
    min_cost = min_fn([D[K, N][[0]], D[K, N][[1]]], gamma=gamma_min)
    return min_cost, D


def batch_dropDTW(
    zx_costs_list, drop_costs_list, softning="prob", exclusive=True, contiguous=True, drop_mode="DropDTW", gamma_min=1
):
    """function to obtain a soft (differentiable version of DTW)
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    # defining the min function
    min_fn = minProb if softning == "prob" else minGamma
    inf = 9999999999

    # pre-processing
    B = len(zx_costs_list)
    Ns, Ks = [], []
    for i in range(B):
        Ki, Ni = zx_costs_list[i].shape
        if drop_mode == "OTAM":
            # add zero row in order to skip in the end
            zero_row = torch.zeros_like(zx_costs_list[i][-1])
            zx_costs_list[i] = torch.cat([zx_costs_list[i], zero_row[None, :]], dim=0)
            Ki += 1

        if Ki >= Ni:
            # in case the number of steps is greater than the number of frames,
            # duplicate every frame and let the drops do the job.
            mult = math.ceil(Ki / Ni)
            zx_costs_list[i] = torch.stack([zx_costs_list[i]] * mult, dim=-1).reshape([Ki, -1])
            drop_costs_list[i] = torch.stack([drop_costs_list[i]] * mult, dim=-1).reshape([-1])
            Ni *= mult
        Ns.append(Ni)
        Ks.append(Ki)
    N, K = max(Ns), max(Ks)

    # preparing padded tables
    padded_cum_drop_costs, padded_drop_costs, padded_zx_costs = [], [], []
    for i in range(B):
        zx_costs = zx_costs_list[i]
        drop_costs = drop_costs_list[i]
        cum_drop_costs = torch.cumsum(drop_costs, dim=0)

        # padding everything to the size of the largest N and K
        row_pad = torch.zeros([N - Ns[i]]).to(zx_costs.device)
        padded_cum_drop_costs.append(torch.cat([cum_drop_costs, row_pad]))
        padded_drop_costs.append(torch.cat([drop_costs, row_pad]))
        multirow_pad = torch.stack([row_pad + inf] * Ks[i], dim=0)
        padded_table = torch.cat([zx_costs, multirow_pad], dim=1)
        rest_pad = torch.zeros([K - Ks[i], N]).to(zx_costs.device) + inf
        padded_table = torch.cat([padded_table, rest_pad], dim=0)
        padded_zx_costs.append(padded_table)

    all_zx_costs = torch.stack(padded_zx_costs, dim=-1)
    all_cum_drop_costs = torch.stack(padded_cum_drop_costs, dim=-1)
    all_drop_costs = torch.stack(padded_drop_costs, dim=-1)

    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, 3, B))  # This corresponds to B 3-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + inf
    for xi in range(1, N + 1):
        if drop_mode == "DropDTW":
            D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[(xi - 1) : xi]
        elif drop_mode == "OTAM":
            D[0, xi] = torch.zeros_like(D[0, xi])
        else:  # drop_mode == 'DTW'
            D[0, xi] = torch.zeros_like(D[0, xi]) + inf

    # obtain dtw table using min_gamma or softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            d_diag, d_left = D[zi - 1, xi - 1][0:1], D[zi, xi - 1][0:1]
            dp_left, dp_up = D[zi, xi - 1][2:3], D[zi - 1, xi][2:3]

            if drop_mode == "DropDTW":
                # positive transition, i.e. matching x_i to z_j
                if contiguous:
                    pos_neighbors = [d_diag, dp_left]
                else:
                    pos_neighbors = [d_diag, d_left]
                if not exclusive:
                    pos_neighbors.append(dp_up)

                Dp = min_fn(pos_neighbors, gamma=gamma_min) + all_zx_costs[z_cost_ind, x_cost_ind]

                # negative transition, i.e. dropping xi
                Dm = d_left + all_drop_costs[x_cost_ind]

                # update final solution matrix
                D_final = min_fn([Dm, Dp], gamma=gamma_min)
            else:
                d_right = D[zi - 1, xi][0:1]
                D_final = Dm = Dp = (
                    min_fn([d_diag, d_left, d_right], gamma=gamma_min) + all_zx_costs[z_cost_ind, x_cost_ind]
                )
            D[zi, xi] = torch.cat([D_final, Dm, Dp], dim=0)

    # Computing the final min cost for the whole batch
    min_costs = []
    for i in range(B):
        Ni, Ki = Ns[i], Ks[i]
        min_cost_i = D[Ki, Ni][0, i]
        min_costs.append(min_cost_i / Ni)

    return min_costs, D


def batch_NW(zx_costs_list, drop_costs_list, softning="prob", gamma_min=1):
    """function to obtain a soft (differentiable version of DTW)
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    # defining the function
    min_fn = minProb if softning == "prob" else minGamma

    # pre-processing
    B = len(zx_costs_list)
    Ns, Ks = [], []
    for i in range(B):
        Ki, Ni = zx_costs_list[i].shape
        if Ki >= Ni:
            # in case the number of steps is greater than the number of frames,
            # duplicate every frame and let the drops do the job.
            mult = math.ceil(Ki / Ni)
            zx_costs_list[i] = torch.stack([zx_costs_list[i]] * mult, dim=-1).reshape([Ki, -1])
            drop_costs_list[i] = torch.stack([drop_costs_list[i]] * mult, dim=-1).reshape([-1])
            Ni *= mult
        Ns.append(Ni)
        Ks.append(Ki)
    N, K = max(Ns), max(Ks)

    # preparing padded tables
    padded_cum_drop_costs, padded_drop_costs, padded_zx_costs = [], [], []
    for i in range(B):
        zx_costs = zx_costs_list[i]
        drop_costs = drop_costs_list[i]
        cum_drop_costs = torch.cumsum(drop_costs, dim=0)

        # padding everything to the size of the largest N and K
        row_pad = torch.zeros([N - Ns[i]]).to(zx_costs.device)
        padded_cum_drop_costs.append(torch.cat([cum_drop_costs, row_pad]))
        padded_drop_costs.append(torch.cat([drop_costs, row_pad]))
        multirow_pad = torch.stack([row_pad + 9999999999] * Ks[i], dim=0)
        padded_table = torch.cat([zx_costs, multirow_pad], dim=1)
        rest_pad = torch.zeros([K - Ks[i], N]).to(zx_costs.device) + 9999999999
        padded_table = torch.cat([padded_table, rest_pad], dim=0)
        padded_zx_costs.append(padded_table)

    all_zx_costs = torch.stack(padded_zx_costs, dim=-1)
    all_cum_drop_costs = torch.stack(padded_cum_drop_costs, dim=-1)
    all_drop_costs = torch.stack(padded_drop_costs, dim=-1)

    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, B))  # This corresponds to B 3-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + 9999999999
    for xi in range(1, N + 1):
        D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[xi - 1]

    # obtain dtw table using min_gamma or softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            match_cost = all_zx_costs[zi - 1, xi - 1]
            drop_cost = all_drop_costs[xi - 1]
            transition_costs = [D[zi - 1, xi - 1] + drop_cost, D[zi - 1, xi] + match_cost, D[zi, xi - 1] + match_cost]
            D[zi, xi] = min_fn(transition_costs, gamma=gamma_min, keepdim=False)

    # Computing the final min cost for the whole batch
    min_costs = []
    for i in range(B):
        Ni, Ki = Ns[i], Ks[i]
        min_cost_i = D[Ki, Ni][i]
        min_costs.append(min_cost_i / Ni)

    return min_costs, D


def batch_double_dropDTW(zx_costs_list, drop_costs_list, exclusive=True, contiguous=True, gamma_min=1):
    """function to obtain a soft (differentiable version of DTW)
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    min_fn = lambda x: minProb(x, gamma=gamma_min)

    # assuming sequences are the same length
    B = len(zx_costs_list)
    N, K = zx_costs_list[0].shape
    cum_drop_costs_list = [torch.cumsum(drop_costs_list[i], dim=0) for i in range(B)]

    all_zx_costs = torch.stack(zx_costs_list, dim=-1)
    all_cum_drop_costs = torch.stack(cum_drop_costs_list, dim=-1)
    all_drop_costs = torch.stack(drop_costs_list, dim=-1)

    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, 4, B))  # This corresponds to B 4-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + all_cum_drop_costs[(zi - 1) : zi]
    for xi in range(1, N + 1):
        D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[(xi - 1) : xi]

    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1, 2, 3]  # zx, z-, -x, --
            diag_neigh_costs = [D[zi - 1, xi - 1][s] for s in diag_neigh_states]

            left_neigh_states = [0, 1]  # zx and z-
            left_neigh_costs = [D[zi, xi - 1][s] for s in left_neigh_states]

            upper_neigh_states = [0, 2]  # zx and -x
            upper_neigh_costs = [D[zi - 1, xi][s] for s in upper_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # DP 0: coming to zx
            neigh_costs_zx = diag_neigh_costs + upper_neigh_costs + left_neigh_costs
            D0 = min_fn(neigh_costs_zx) + all_zx_costs[z_cost_ind, x_cost_ind]

            # DP 1: coming to z-
            neigh_costs_z_ = left_neigh_costs
            D1 = min_fn(neigh_costs_z_) + all_drop_costs[x_cost_ind]

            # DP 2: coming to -x
            neigh_costs__x = upper_neigh_costs
            D2 = min_fn(neigh_costs__x) + all_drop_costs[z_cost_ind]

            # DP 3: coming to --
            costs___ = [d + all_drop_costs[z_cost_ind] * 2 for d in diag_neigh_costs] + [
                D[zi, xi - 1][3] + all_drop_costs[x_cost_ind],
                D[zi - 1, xi][3] + all_drop_costs[z_cost_ind],
            ]
            D3 = min_fn(costs___)

            D[zi, xi] = torch.cat([D0, D1, D2, D3], dim=0)

    # Computing the final min cost for the whole batch
    min_costs = []
    for i in range(B):
        min_cost_i = min_fn(D[K, N][:, i])
        min_costs.append(min_cost_i / N)
    return min_costs, D


def prob_min(tensors, gamma_min):
    if len(tensors) > 1:
        stacked = torch.cat(tensors, dim=-1)
    else:
        stacked = tensors[0]
    probs = F.softmax(-stacked / gamma_min, dim=-1)
    return (stacked * probs).sum(-1)


def drop_dtw_machine(zx_costs, drop_costs, gamma_min=1, exclusive=True, contiguous=True):
    K, N = zx_costs.shape
    dev = zx_costs.device
    flipped_costs = torch.flip(zx_costs, [0])  # flip the cost matrix upside down
    cum_drop_costs = torch.cumsum(drop_costs, dim=-1)

    # initialize first two contr diagonals
    inf = torch.tensor([9999999999], device=dev, dtype=zx_costs.dtype)
    diag_pp = torch.zeros([1, 2], device=dev)  # diag at i-2
    diag_p_col = torch.ones([1, 2], device=dev) * inf
    diag_p_row = torch.stack([inf, cum_drop_costs[[0]]], -1)
    diag_p = torch.cat([diag_p_row, diag_p_col], 0)  # diag at i-1

    for i in range(K + N - 1):
        size = diag_p.size(0) - 1
        pp_start = max(0, diag_pp.size(0) - diag_p.size(0))
        neigh_up, neigh_left, neigh_diag = diag_p[:-1], diag_p[1:], diag_pp[pp_start : (pp_start + size)]
        neigh_up_pos, neigh_left_pos = neigh_up[:, [0]], neigh_left[:, [0]]

        # define match and drop cost vectors
        match_costs_diag = torch.flip(torch.diag(flipped_costs, i + 1 - K), [-1])
        d_start, d_end = max(1 - K + i, 0), min(i, N - 1) + 1
        drop_costs_diag = torch.flip(drop_costs[d_start:d_end], [-1])

        # update positive and negative tables -> compute new diagonal
        pos_neighbors = [neigh_diag, neigh_left_pos] if contiguous else [neigh_diag, neigh_left]
        if not exclusive:
            pos_neighbors.append(neigh_up_pos)
        diag_pos = prob_min(pos_neighbors, gamma_min) + match_costs_diag
        diag_neg = prob_min([neigh_left], gamma_min) + drop_costs_diag
        diag = torch.stack([diag_pos, diag_neg], -1)

        # add the initialization values on the ends of diagonal if needed
        if i < N - 1:
            # fill in 0th row with [drop_cost, inf]
            pad = torch.stack([inf, cum_drop_costs[[i + 1]]], -1)
            diag = torch.cat([pad, diag])
        if i < K - 1:
            # fill in 0th col with [inf, inf]
            pad = torch.stack([inf, inf], -1)
            diag = torch.cat([diag, pad])

        diag_pp = diag_p
        diag_p = diag
    assert (diag.size(0) == 1) and (diag.size(1) == 2), f"Last diag shape is {diag.shape} instead of [1, 2]"

    cost = prob_min(diag, gamma_min)
    return cost


def batch_drop_dtw_machine(zx_costs_list, drop_costs_list, gamma_min=1, exclusive=True, contiguous=True):
    dev, dtype = zx_costs_list[0].device, zx_costs_list[0].dtype
    inf = torch.tensor([9999999999], device=dev, dtype=dtype)
    B = len(zx_costs_list)

    Ns, Ks = [], []
    for i in range(B):
        Ki, Ni = zx_costs_list[i].shape
        if Ki >= Ni:
            # in case the number of steps is greater than the number of frames,
            # duplicate every frame and let the drops do the job.
            mult = math.ceil(Ki / Ni)
            zx_costs_list[i] = torch.stack([zx_costs_list[i]] * mult, dim=-1).reshape([Ki, -1])
            drop_costs_list[i] = torch.stack([drop_costs_list[i]] * mult, dim=-1).reshape([-1])
            Ni *= mult
        Ns.append(Ni)
        Ks.append(Ki)
    # N, K = max(Ns) + 1, max(Ks) + 1
    N, K = max(Ns), max(Ks)

    # transform endpoints into diagonal coordinates
    Ds, Rs = torch.zeros(B).to(dev).to(int), torch.zeros(B).to(dev).to(int)
    for i, (Ki, Ni) in enumerate(zip(Ks, Ns)):
        Ds[i] = Ki + Ni - 2
        Rs[i] = min(Ds[i] + 2, N) - Ni
    Ds_orig, Rs_orig = copy(Ds), copy(Rs)

    # special padding of costs to ensure that the path goest through the endpoint
    all_zx_costs = []
    for i, c in enumerate(zx_costs_list):
        c_inf_frame = F.pad(c, [0, 1, 0, 1], value=inf.item())
        mask = torch.ones_like(c_inf_frame)
        mask[-1, -1] = 0
        # c_pad = F.pad(c_inf_frame * mask, [0, K - c.shape[0] - 1, 0, N - c.shape[1] - 1])
        c_pad = F.pad(c_inf_frame * mask, [0, N - c.shape[1] - 1, 0, K - c.shape[0] - 1])
        all_zx_costs.append(c_pad)
    all_zx_costs = torch.stack(all_zx_costs, 0)

    # all_zx_costs = torch.stack([F.pad(c, [0, K - c.shape[0], 0, N - c.shape[1]]) for c in zx_costs_list], 0)
    all_drop_costs = torch.stack([F.pad(c, [0, N - c.shape[0]], value=inf.item()) for c in drop_costs_list], 0)
    all_cum_drop_costs = torch.stack(
        [F.pad(torch.cumsum(c, -1), [0, N - c.shape[0]], value=inf.item()) for c in drop_costs_list], 0
    )
    flipped_costs = torch.flip(all_zx_costs, [1])  # flip the cost matrix upside down

    # initialize first two contr diagonals
    batch_inf = torch.stack([inf] * B, 0)
    diag_pp = torch.zeros([B, 1, 2], device=dev)  # diag at i-2
    diag_p_col = torch.ones([B, 1, 2], device=dev) * batch_inf[..., None]
    diag_p_row = torch.stack([batch_inf, all_cum_drop_costs[:, [0]]], -1)
    diag_p = torch.cat([diag_p_row, diag_p_col], 1)  # diag at i-1

    min_costs = torch.zeros(B).to(dtype=dtype).to(device=dev)
    for d in range(K + N - 1):
        size = diag_p.size(1) - 1
        pp_start = max(0, diag_pp.size(1) - diag_p.size(1))
        neigh_up, neigh_left, neigh_diag = diag_p[:, :-1], diag_p[:, 1:], diag_pp[:, pp_start : (pp_start + size)]
        neigh_up_pos, neigh_left_pos = neigh_up[..., [0]], neigh_left[..., [0]]

        # define match and drop cost vectors
        match_costs_diag = torch.stack(
            [torch.flip(torch.diag(flipped_costs[j], d + 1 - K), [-1]) for j in range(flipped_costs.size(0))], 0
        )

        d_start, d_end = max(1 - K + d, 0), min(d, N - 1) + 1
        drop_costs_diag = torch.flip(all_drop_costs[:, d_start:d_end], [-1])

        # update positive and negative tables -> compute new diagonal
        pos_neighbors = [neigh_diag, neigh_left_pos] if contiguous else [neigh_diag, neigh_left]
        if not exclusive:
            pos_neighbors.append(neigh_up_pos)
        diag_pos = prob_min(pos_neighbors, gamma_min) + match_costs_diag
        diag_neg = prob_min([neigh_left], gamma_min) + drop_costs_diag
        diag = torch.stack([diag_pos, diag_neg], -1)

        # add the initialization values on the ends of diagonal if needed
        if d < N - 1:
            # fill in 0th row with [drop_cost, inf]
            pad = torch.stack([batch_inf, all_cum_drop_costs[:, [d + 1]]], -1)
            diag = torch.cat([pad, diag], 1)
        if d < K - 1:
            # fill in 0th col with [inf, inf]
            pad = torch.stack([batch_inf, batch_inf], -1)
            diag = torch.cat([diag, pad], 1)

        diag_pp = diag_p
        diag_p = diag

        # process answers
        if (Ds == d).any():
            mask, orig_mask = Ds == d, Ds_orig == d
            bs, rs = torch.nonzero(mask, as_tuple=False)[:, 0], Rs[mask]
            min_costs[orig_mask] = min_costs[orig_mask] + prob_min([diag[bs, rs]], gamma_min)

            diag, diag_p, diag_pp, Ds, Rs, flipped_costs, all_drop_costs, all_cum_drop_costs, batch_inf = [
                t[~mask]
                for t in [diag, diag_p, diag_pp, Ds, Rs, flipped_costs, all_drop_costs, all_cum_drop_costs, batch_inf]
            ]
            if torch.numel(Ds) == 0:
                break

    # costs = prob_min([diag], gamma_min)
    costs_norm = min_costs / torch.tensor(Ns).to(dev)
    return costs_norm
