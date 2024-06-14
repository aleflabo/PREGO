import os
import json
import torch
import numpy as np
import networkx as nx
from copy import deepcopy

from data.loader import LMDB_Folder_Dataset
from data.data_utils import dict2tensor
from dp.graph_utils import remove_nodes_from_graph
from models.text_encoder import get_text_encoder
from paths import CT_PATH


device = "cpu"
text_encoder = get_text_encoder(device)


def init_dataset(features="mil"):
    global ct_dataset, name_to_idx

    if features == "mil":
        lmdb_path = os.path.join(CT_PATH, "lmdb")
        ct_dataset = LMDB_Folder_Dataset(lmdb_path, split="val", transform=dict2tensor)
        name_to_idx = {"_".join(s["name"].split("_")[1:]): idx for idx, s in enumerate(ct_dataset)}
    else:
        lmdb_path = os.path.join(CT_PATH, "lmdb_univl2")
        ct_dataset = LMDB_Folder_Dataset(lmdb_path, split="val", transform=dict2tensor)
        name_to_idx = {s["name"]: idx for idx, s in enumerate(ct_dataset)}
    return name_to_idx, ct_dataset


# graph annotations
def init_graph(type="manual"):
    global graph_annots

    if type == "manual":
        graph_path = os.path.join(CT_PATH, "graphs", "manual.json")
    elif type == "learned_parser":
        graph_path = os.path.join(CT_PATH, "graphs", "learned_parser.json")
    elif type == "rule_parser":
        graph_path = os.path.join(CT_PATH, "graphs", "rule_parser.json")

    with open(graph_path, "r") as f:
        graph_annots = json.load(f)
        graph_annots = {s["recipe_type"]: s for s in graph_annots}

    if type == "learned_parser":
        for k in graph_annots:
            graph_annots[k]["recipe"].pop("edges")
            graph_annots[k]["recipe"]["edges"] = graph_annots[k]["predicted_edges"]
        print("removed edges")
    return graph_annots


def get_orig_sample(name):
    sample = ct_dataset[name_to_idx[name]]
    sample["vid_id"] = name
    try:
        sample["step_sentences"] = [ct_dataset.get_step_description(s.item()) for s in sample["step_ids"]]
    except:
        sample["step_sentences"] = [ct_dataset.get_step_description(s) for s in sample["step_ids"]]
    sample["url"] = "https://www.youtube.com/watch?v=" + name

    gt_assignment = -torch.ones((sample["frame_features"].shape[0],), dtype=int)
    for s, step_id in enumerate(sample["step_ids"]):
        gt_assignment[sample["step_starts"][s] : sample["step_ends"][s] + 1] = step_id
    sample["gt_seg"] = gt_assignment
    return sample


def get_graph_sample(name, remove_absent_steps=False):
    sample = deepcopy(get_orig_sample(name))
    try:
        recipe_id = str(sample["cls"].item())
    except:
        recipe_id = str(sample["cls"])
    cls_dataset = [cd for cd in ct_dataset.cls_datasets if str(cd.cls_id) == recipe_id][0]
    cd_cls_ids = cls_dataset.step_descriptions.keys()
    cd_cls_ids_2_base_ids = {c: i for i, c in enumerate(cd_cls_ids)}

    new_step_ids = [cd_cls_ids_2_base_ids[s.item()] for s in sample["step_ids"]]
    sample["step_ids"] = torch.tensor(np.array(new_step_ids))
    sample["orig_step_ids"] = sample["step_ids"]

    # taking the instruction annots from the file
    recipe_instructions = graph_annots[recipe_id]["recipe"]["instructions"]
    recipe_edges = graph_annots[recipe_id]["recipe"]["edges"]
    recipe_2_caption = dict(graph_annots[recipe_id]["recipe"]["recipe_2_caption"])
    sample["recipe_2_caption"] = recipe_2_caption

    cap_2_rep = {c.item(): -1 for c in sample["orig_step_ids"]}
    for r, c in sample["recipe_2_caption"].items():
        cap_2_rep[c] = r
    sample["caption_2_recipe"] = cap_2_rep
    sample["node_ids"] = [sample["caption_2_recipe"][s.item()] for s in sample["orig_step_ids"]]

    # deriving the gt segmentation
    num_frames = sample["frame_features"].shape[0]
    gt_assignment = -torch.ones((num_frames,), dtype=int)
    for s, step_id in enumerate(sample["step_ids"]):
        gt_assignment[sample["step_starts"][s] : sample["step_ends"][s] + 1] = step_id
    sample["gt_seg"] = gt_assignment

    # convert graph steps into gt steps
    new_K = len(recipe_instructions)
    sample["step_ids"] = torch.arange(new_K)
    sample["step_features"] = text_encoder(recipe_instructions)
    gt_step_ids = [recipe_2_caption[s.item()] for s in sample["step_ids"]]
    sample["gt_step_ids"] = torch.tensor(np.array(gt_step_ids))
    sample["num_steps"] = torch.tensor(sample["step_features"].shape[0])
    sample["step_sentences"] = recipe_instructions

    G = nx.DiGraph()
    [G.add_node(i) for i in range(len(recipe_instructions))]
    for edge in recipe_edges:
        G.add_edge(edge[0], edge[1])

    if remove_absent_steps:
        present_nodes = [r for r, c in recipe_2_caption.items() if c > -1]
        absent_nodes = [r for r, c in recipe_2_caption.items() if c == -1]
        all_to_present = {p: i for i, p in enumerate(present_nodes)}
        sample["step_ids"] = torch.arange(len(present_nodes))
        sample["gt_step_ids"] = sample["gt_step_ids"][present_nodes]
        sample["step_features"] = sample["step_features"][present_nodes]
        sample["num_steps"] = torch.tensor(len(present_nodes))
        sample["step_sentences"] = [sample["step_sentences"][i] for i in present_nodes]
        sample["recipe_2_caption"] = {all_to_present[r]: recipe_2_caption[r] for r in present_nodes}

        G = remove_nodes_from_graph(G, absent_nodes, relabel=False)
        nodes, edges = list(G.nodes()), list(G.edges())
        G = nx.DiGraph()
        [G.add_node(all_to_present[i]) for i in nodes]
        [G.add_edge(all_to_present[i], all_to_present[j]) for i, j in edges]

    sample["graph"] = G

    return sample


def get_inverse_sample(orig_sample, graph_sample):
    inverse_sample = deepcopy(orig_sample)
    orig_step_ids = graph_sample["orig_step_ids"].numpy()
    caption_2_recipe = {c: -1 for c in orig_step_ids}
    for r, c in graph_sample["recipe_2_caption"].items():
        caption_2_recipe[c] = r
    orig_ids_in_recipes = [caption_2_recipe[c] for c in orig_step_ids]
    full_lookup_mat = torch.cat([graph_sample["step_features"], torch.zeros([1, 512])], dim=0)
    inverse_sample["step_features"] = full_lookup_mat[orig_ids_in_recipes]

    # change the gt_segmentation to mapped values
    for i, v in enumerate(graph_sample["gt_seg"]):
        if v > -1:
            inverse_sample["gt_seg"][i] = caption_2_recipe[v.item()]

    # change the gt_segmentation to mapped values
    for i, v in enumerate(graph_sample["orig_step_ids"]):
        inverse_sample["step_ids"][i] = caption_2_recipe[v.item()]

    return inverse_sample


def map_segmentation_to_graph_nodes(labels, recipe2caption):
    mapped_labels = -np.ones_like(labels)
    for rec_id, cap_id in recipe2caption.items():
        mapped_labels[labels == rec_id] = cap_id
    return mapped_labels
