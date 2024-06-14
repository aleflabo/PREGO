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
from paths import YC_PATH

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
text_encoder = get_text_encoder(device)


lmdb_path = os.path.join(YC_PATH, "lmdb")
yc_dataset = LMDB_Folder_Dataset(lmdb_path, split="val", transform=dict2tensor)
name_to_idx = {"_".join(s["name"].split("_")[1:]): idx for idx, s in enumerate(yc_dataset)}
all_names = [s for s in name_to_idx.keys() if s not in ["bmZB3aszZlA", "OEfzgobszUA", "T_o_T3LEYLY"]]

annotation_json = os.path.join(YC_PATH, "youcookii_annotations_trainval.json")
with open(annotation_json, "r") as f:
    annots = json.load(f)["database"]
annots = {k: annots[k] for k in name_to_idx.keys() if k in all_names}

# graph annotations
graph_path = os.path.join(YC_PATH, "graphs", "youcookii_annotations_val_with_sent_edges.json")
with open(graph_path, "r") as f:
    graph_annots = json.load(f)
graph_annots = {s["vid_id"]: s for s in graph_annots if s["vid_id"] in all_names}

# recipe annotations
recipe_path = os.path.join(YC_PATH, "graphs", "yc2_recipes_full.json")
with open(recipe_path, "r") as f:
    recipes_annots = json.load(f)
recipes_annots = {s["vid_id"]: s for s in recipes_annots if s["vid_id"] in all_names}


def get_orig_sample(name):
    sample = yc_dataset[name_to_idx[name]]
    sample["vid_id"] = name
    sample["step_sentences"] = [a["sentence"] for a in annots[name]["annotations"]]
    sample["url"] = annots[name]["video_url"]
    return sample


def get_graph_sample(name):
    orig_sample = get_orig_sample(name)
    sample = deepcopy(orig_sample)
    step_sentences = [s.replace(",", "") for s in graph_annots[name]["sentences"]]
    if step_sentences is not None:
        sample["step_sentences"] = step_sentences
        sample["step_features"] = text_encoder(step_sentences)

    K_new = len(step_sentences)
    sample["num_steps"] = torch.tensor(K_new)
    sample["step_ids"] = torch.arange(K_new)
    sample["step_starts_sec"], sample["step_ends_sec"] = torch.zeros([K_new]), torch.zeros([K_new])
    sample["step_starts"], sample["step_ends"] = torch.zeros([K_new]).to(int), torch.zeros([K_new]).to(int)
    for i, segment in enumerate(graph_annots[name]["segments"]):
        sample["step_starts_sec"][i], sample["step_ends_sec"][i] = segment
        sample["step_starts"][i], sample["step_ends"][i] = [s // 3.2 for s in segment]

    # build a graph
    G = nx.DiGraph()
    [G.add_node(i) for i in range(len(step_sentences))]
    for edge in graph_annots[name]["edges"]:
        G.add_edge(edge[0], edge[1])

    # sink_node = edge[1]
    # handle disconnected nodes
    # for node in G:
    #     if len(list(G.successors(node))) == 0 and node != sink_node:
    #         G.add_edge(node, sink_node)

    sample["graph"] = G
    return sample


def get_recipe_sample(name, merge=True, filter=True):
    graph_sample = get_graph_sample(name)
    sample = deepcopy(graph_sample)
    sample["orig_step_sentences"] = deepcopy(graph_sample["step_sentences"])
    sample["step_sentences"] = ["" for _ in range(sample["step_features"].shape[0])]
    sample["step_features"] *= 0

    recipe = deepcopy(recipes_annots[name]["recipe"])
    sample["recipe"] = recipe
    instructions = [s.replace(",", "").replace(".", "") for s in recipe["instructions"]]
    instruction_features = text_encoder(instructions)

    caption_2_recipes = dict()
    for rec_id, cap_id in recipe["recipe_2_caption"]:
        caption_2_recipes[cap_id] = caption_2_recipes.get(cap_id, []) + [rec_id]

    present_steps = np.array(list(caption_2_recipes.keys()))
    present_steps = present_steps[present_steps >= 0]
    missing_steps = [s for s in np.arange(len(sample["step_sentences"])) if s not in present_steps]
    for i in present_steps:
        rec_ids = caption_2_recipes[i]
        sample["step_features"][i] = instruction_features[rec_ids].mean()
        sample["step_sentences"][i] = ". ".join([instructions[rid] for rid in rec_ids])

    if filter:
        # only present steps remain in the annotations
        for key in graph_sample.keys():
            if key.startswith("step"):
                if isinstance(sample[key], list):
                    sample[key] = [sample[key][i] for i in present_steps]
                else:
                    sample[key] = sample[key][present_steps]
        sample["num_steps"] = torch.tensor(len(sample["step_sentences"]))
        sample["graph"] = remove_nodes_from_graph(sample["graph"], missing_steps, relabel=True)

    return sample
