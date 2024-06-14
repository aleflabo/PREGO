import torch
import numpy as np
import torch.nn.functional as F
import networkx as nx
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor


class Node:
    def __init__(self, node_id, parents):
        self.node_id = node_id
        self.parents = parents
        self.neighbors_up = set()

    def __repr__(self):
        return f"N{self.node_id}: p" + "_".join(str(p.node_id) for p in self.parents)

    def push_down_neighbors(self, neighbors_up):
        self.neighbors_up = self.neighbors_up.union(neighbors_up)
        if len(self.parents) == 0:
            self.neighbors_down = set()
        else:
            neigh = neighbors_up.union({self.node_id})
            all_neighbors_down = [p.push_down_neighbors(neigh) for p in self.parents]
            self.neighbors_down = set().union(*all_neighbors_down)

        return self.neighbors_down.union({self.node_id})

    def get_thread(self):
        return self.neighbors_up.union(self.neighbors_down)

    def get_parallel_nodes(self, all_nodes):
        return all_nodes - self.get_thread().union({self.node_id})


def create_dag(parents_dict):
    # assumes parents dict is tomologically sorted
    all_nodes = set(list(parents_dict.keys()))
    sinks = all_nodes
    for parents in parents_dict.values():
        sinks = sinks - set(parents)

    node_dict = {}
    for node_id, parents in parents_dict.items():
        parent_nodes = [node_dict[p] for p in parents]
        node_dict[node_id] = Node(node_id, parent_nodes)
    return node_dict, [node_dict[s] for s in sinks]


def topological_sort(dag):
    sorted_dag = dict()
    tmp_dag = {k: v for k, v in dag.items()}
    while len(tmp_dag) > 0:
        for node, parents in list(tmp_dag.items()):
            parents = [p for p in parents if p not in sorted_dag]
            if len(parents) == 0:
                sorted_dag[node] = dag[node]
                tmp_dag.pop(node)
            else:
                tmp_dag[node] = parents
    return sorted_dag


def process_dag(dag):
    sorted_dag = topological_sort(dag)
    dag_nodes, sink_nodes = create_dag(sorted_dag)
    all_nodes = set(list(dag_nodes.keys()))
    for s in sink_nodes:
        s.push_down_neighbors(set())

    thread_dict = {nid: n.get_thread() for nid, n in dag_nodes.items()}
    parallel_dict = {nid: n.get_parallel_nodes(all_nodes) for nid, n in dag_nodes.items()}
    return sorted_dag, thread_dict, parallel_dict


def compute_dag_costs(
    sample,
    parallel_dict,
    gamma_xz=10,
    drop_cost_type="raw",
    keep_percentile=0.3,
    l2_normalize=False,
):
    """This function computes pairwise match and individual drop costs used in Drop-DTW

    Parameters
    __________

    sample: dict
        sample dictionary
    drop_cost_type: str
        The type of drop cost definition, i.g., learnable or logits percentile.
    keep_percentile: float in [0, 1]
        if drop_cost_type == 'logit', defines drop (keep) cost threshold as logits percentile
    l2_normalize: bool
        wheather to normalize clip and step features before computing the costs
    """

    labels = sample["step_ids"]
    step_features, frame_features = sample["step_features"], sample["frame_features"]
    if l2_normalize:
        frame_features = F.normalize(frame_features, p=2, dim=1)
        step_features = F.normalize(step_features, p=2, dim=1)
    sim = step_features @ frame_features.T / gamma_xz

    unique_labels, unique_index, unique_inverse_index = np.unique(
        labels.detach().cpu().numpy(), return_index=True, return_inverse=True
    )
    unique_sim = sim[unique_index]

    if drop_cost_type in ["logit", "raw"]:
        k = max([1, int(torch.numel(unique_sim) * keep_percentile)])
        baseline_logit = torch.topk(unique_sim.reshape([-1]), k).values[-1].detach()
        drop_logits = baseline_logit.repeat([1, unique_sim.shape[1]])  # making it of shape [1, N]
    else:
        assert False, f"No such drop mode {drop_cost_type}"

    all_drop_logits = []
    node_id2idx = {n_id: idx for idx, n_id in enumerate(parallel_dict.keys())}
    for par_nodes in parallel_dict.values():
        if len(par_nodes) > 0:
            parallel_idxs = [node_id2idx[n_id] for n_id in par_nodes]
            competing_match_sim = sim[parallel_idxs].max(0).values
            competing_match_sim = competing_match_sim.reshape([-1, sim.shape[1]])
            step_drop_logits = torch.stack([competing_match_sim, drop_logits], 0).max(0).values
        else:
            step_drop_logits = drop_logits
        all_drop_logits.append(step_drop_logits)
    drop_logits_mat = torch.cat(all_drop_logits, 0)
    assert drop_logits_mat.shape == sim.shape, "Shape mismatch"
    # sims_ext = torch.cat([unique_sim, baseline_logits], dim=0)

    # unique_softmax_sims = torch.nn.functional.softmax(sims_ext / gamma_xz, dim=0)
    # unique_softmax_sim, drop_probs = unique_softmax_sims[:-1], unique_softmax_sims[-1]
    # matching_probs = unique_softmax_sim[unique_inverse_index]
    # zx_costs = -torch.log(matching_probs + 1e-5)
    # drop_costs = -torch.log(drop_probs + 1e-5)
    return -sim, -drop_logits_mat, -drop_logits[0]


def compute_metadag_costs(
    sample,
    idx2node,
    gamma_xz=10,
    drop_cost_type="raw",
    keep_percentile=0.3,
    l2_normalize=False,
    do_logsoftmax=False,
    object="step",
):
    """This function computes pairwise match and individual drop costs used in Drop-DTW

    Parameters
    __________

    sample: dict
        sample dictionary
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
    sim = step_features @ frame_features.T / gamma_xz

    unique_labels, unique_index, unique_inverse_index = np.unique(
        labels.detach().cpu().numpy(), return_index=True, return_inverse=True
    )
    unique_sim = sim[unique_index]

    if drop_cost_type in ["logit", "raw"]:
        k = max([1, int(torch.numel(unique_sim) * keep_percentile)])
        baseline_logit = torch.topk(unique_sim.reshape([-1]), k).values[-1].detach()
        drop_logits = baseline_logit.repeat([1, unique_sim.shape[1]])  # making it of shape [1, N]
    else:
        assert False, f"No such drop mode {drop_cost_type}"

    if do_logsoftmax:
        sims_ext = torch.cat([unique_sim, drop_logits], dim=0)
        unique_softmax_sims = torch.nn.functional.softmax(sims_ext / gamma_xz, dim=0)
        unique_softmax_sim, drop_probs = unique_softmax_sims[:-1], unique_softmax_sims[-1]
        matching_probs = unique_softmax_sim[unique_inverse_index]
        meta_zx_costs = -torch.log(matching_probs + 1e-5)
        drop_costs = -torch.log(drop_probs + 1e-5).squeeze()
    else:
        meta_zx_costs = -sim
        drop_costs = -drop_logits.squeeze()

    active_nodes = np.array([int(float(v.split(",")[0])) for v in idx2node.values()])
    active_nodes -= active_nodes.min()  # offset to 0
    zx_costs = meta_zx_costs[active_nodes, :]
    assert zx_costs.shape[0] == len(idx2node), "Shape mismatch"
    return zx_costs, drop_costs


def dag2vid(zx_costs, drop_costs, dag, exclusive=True, contiguous=True, return_labels=False):
    """DAG-match algorithm that allows drop only from one (video) side. See Algorithm 1 in the paper.

    Parameters
    ----------
    zx_costs: np.ndarray [K, N]
        pairwise match costs between K steps and N video clips
    drop_costs: np.ndarray [N]
        drop costs for each clip
    dag: dict of lists
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
    K, N = zx_costs.shape

    # prepare the list of possible states to transition from
    prev_states_dict = dict()
    for node, parents in dag.items():
        if len(parents) == 0:
            prev_states_dict[node + 1] = [0]
        else:
            prev_states_dict[node + 1] = [s + 1 for s in parents]

    # initialize solutin matrices

    # the 2 last dimensions correspond to different states.
    # State (dim) 0 - x is matched; State 1 - x is dropped
    D = np.zeros([K + 1, N + 1, 2])

    D[1:, 0, :] = np.inf  # no drops in z in any state
    D[0, 1:, 0] = np.inf  # no drops in x in state 0, i.e. state where x is matched
    D[0, 1:, 1] = np.cumsum(drop_costs)  # drop costs initizlization in state 1

    # initialize path tracking info for each state
    P = dict()
    for xi in range(1, N + 1):
        P[(0, xi, 1)] = [(0, xi - 1, 1)]

    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # selecting the minimum cost transition (between pos and neg) for each preceeding state
            prev_states_min = []
            for pz in prev_states_dict[zi]:
                if D[pz, xi - 1, 0] > D[pz, xi - 1, 1]:
                    prev_states_min.append((pz, xi - 1, 1))
                else:
                    prev_states_min.append((pz, xi - 1, 0))
            prev_total_cost = sum([D[s] for s in prev_states_min])

            cur_states = [(zi, xi - 1, s) for s in [0, 1]]
            cur_costs = [D[s] for s in cur_states]

            cur_pos_states = [cur_states[0]]
            cur_pos_cost = D[cur_pos_states[0]]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # state 0: x is kept
            match_cost = zx_costs[z_cost_ind, x_cost_ind]
            if cur_pos_cost < prev_total_cost:
                D[zi, xi, 0] = cur_pos_cost + match_cost
                P[(zi, xi, 0)] = cur_pos_states
            else:
                D[zi, xi, 0] = prev_total_cost + match_cost
                P[(zi, xi, 0)] = prev_states_min

            # state 1: x is dropped
            costs_neg = np.array(cur_costs) + drop_costs[x_cost_ind]
            opt_ind_neg = np.argmin(costs_neg)
            D[zi, xi, 1] = costs_neg[opt_ind_neg]
            P[(zi, xi, 1)] = [cur_states[opt_ind_neg]]

    cur_state = D[K, N, :].argmin()

    # backtracking the solution
    full_costs = np.concatenate([drop_costs[None, :], zx_costs], axis=0)
    labels_mat = np.zeros([K + 1, N], dtype=int)
    parents = [(K, N, cur_state)]
    visited_xi = []
    while len(parents) > 0:
        zi, xi, cur_state = parents.pop(0)
        if xi > 0:
            labels_mat[zi * (cur_state == 0), xi - 1] = 1
            parents.extend(P[(zi, xi, cur_state)])
        visited_xi.append(xi)
    # print("visited_xi:", visited_xi)
    full_costs[~labels_mat.astype(bool)] = np.inf
    labels = full_costs.argmin(0) - 1
    min_cost = full_costs[labels, range(N)].sum()
    return min_cost, labels_mat, labels


def metadag2vid(zx_costs, drop_costs, metadag, idx2node, contiguous=True, return_meta_labels=False):
    """DAG-match algorithm that allows drop only from one (video) side. See Algorithm 1 in the paper.

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
    K, N = zx_costs.shape

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

    # initialize solutin matrices

    # the 2 last dimensions correspond to different states.
    D = np.zeros([K + 1, N + 1, 2])

    D[1:, 0, :] = np.inf  # no drops in z in any state
    D[0, 1:, 0] = np.inf  # no drops in x in state 0, i.e. state where x is matched
    D[0, 1:, 1] = np.cumsum(drop_costs)  # drop costs initizlization in state 1

    # initialize path tracking info for each state
    P = dict()
    for xi in range(1, N + 1):
        P[(0, xi, 1)] = (0, xi - 1, 1)

    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # selecting the minimum cost transition (between pos and neg) for each preceeding state
            prev_states_min = []
            for pz in prev_states_dict[zi]:
                if D[pz, xi - 1, 0] > D[pz, xi - 1, 1]:
                    prev_states_min.append((pz, xi - 1, 1))
                else:
                    prev_states_min.append((pz, xi - 1, 0))
            prev_costs = [D[s] for s in prev_states_min]
            argmin_prev_costs = np.array(prev_costs).argmin()
            min_prev_cost = prev_costs[argmin_prev_costs]
            best_prev_state = prev_states_min[argmin_prev_costs]

            cur_states = [(zi, xi - 1, s) for s in [0, 1]]
            cur_costs = [D[s] for s in cur_states]

            cur_pos_states = cur_states[:1] if contiguous else cur_states
            cur_pos_costs = [D[s] for s in cur_pos_states]
            argmin_cur = np.array(cur_pos_costs).argmin()
            cur_pos_state = cur_pos_states[argmin_cur]
            cur_pos_cost = cur_pos_costs[argmin_cur]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # state 0: x is kept
            match_cost = zx_costs[z_cost_ind, x_cost_ind]
            if cur_pos_cost < min_prev_cost:
                D[zi, xi, 0] = cur_pos_cost + match_cost
                P[(zi, xi, 0)] = cur_pos_state
            else:
                D[zi, xi, 0] = min_prev_cost + match_cost
                P[(zi, xi, 0)] = best_prev_state

            # state 1: x is dropped
            costs_neg = np.array(cur_costs) + drop_costs[x_cost_ind]
            opt_ind_neg = np.argmin(costs_neg)
            D[zi, xi, 1] = costs_neg[opt_ind_neg]
            P[(zi, xi, 1)] = cur_states[opt_ind_neg]

    cur_state = D[K, N, :].argmin()

    # backtracking the solution
    labels = np.zeros([N], dtype=int)
    meta_labels = [-1 for _ in range(N)]
    parents = [(K, N, cur_state)]
    while len(parents) > 0:
        zi, xi, cur_state = parents.pop(0)
        if xi > 0:
            meta_node_id = idx2node[zi - 1] if zi > 0 else -1
            meta_labels[xi - 1] = meta_node_id
            label = int(float(meta_node_id.split(",")[0])) if zi > 0 else -1
            labels[xi - 1] = label if cur_state == 0 else -1
            parents.append(P[(zi, xi, cur_state)])
    min_cost = D[K, N].min()

    if return_meta_labels:
        return min_cost, labels, meta_labels
    else:
        return min_cost, labels


def generate_metagraph(G):
    # add global sink to the graph
    presink_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    sink = 999
    G.add_node(sink)
    for node in presink_nodes:
        G.add_edge(node, sink)

    meta_G = nx.DiGraph()
    active_sink = str(sink)
    meta_G.add_node(active_sink)
    queue = [active_sink]
    while queue:
        state = queue.pop(0)
        components = [int(float(n)) for n in state.split(",")]
        active_node = components[0]
        parents = list(G.pred[active_node])
        new_components = parents + components[1:]
        for i in range(len(new_components)):
            new_active_node = new_components[i]
            par_nodes = new_components[0:i] + new_components[i + 1 :]

            # perform reduction
            par_nodes = [n for n in par_nodes if n != new_active_node]
            par_nodes = sorted(list(set(par_nodes)))

            new_state_components = [new_active_node] + par_nodes
            new_state = ",".join(str(c) for c in new_state_components)

            # checking for feasibility, i.e. like checking tokens
            feasible = True
            for par_node in par_nodes:
                if lowest_common_ancestor(G, new_active_node, par_node) == new_active_node:
                    # active node is an ancestor of some already matched nodes -> impossible -> reject candidate
                    feasible = False
            if not feasible:
                continue

            # add new state to meta_G
            if new_state not in meta_G.nodes():
                meta_G.add_node(new_state)

            if (new_state, state) not in meta_G.edges():
                meta_G.add_edge(new_state, state)
                if new_state not in queue:
                    queue.append(new_state)

    # remove sink from G
    for node in presink_nodes:
        G.remove_edge(node, sink)
    G.remove_node(sink)

    # remove sink from meta_G
    sink = str(sink)
    for node in list(meta_G.predecessors(sink)):
        meta_G.remove_edge(node, sink)
    meta_G.remove_node(sink)

    return meta_G


def remove_nodes_from_graph(G, nodes_to_remove, relabel=True):
    for node in sorted(nodes_to_remove):
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))

        for pred in predecessors:
            G.remove_edge(pred, node)

        for succ in successors:
            G.remove_edge(node, succ)

        for pred in predecessors:
            for succ in successors:
                G.add_edge(pred, succ)
        G.remove_node(node)

    if relabel:
        mapping = {n: i for i, n in enumerate(G)}
        G = nx.relabel_nodes(G, mapping)
    return G
