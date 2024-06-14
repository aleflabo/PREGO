import statistics
import sys

import numpy as np
from coin_loader import COINDataset
from construct_graph import TaskGraph
from crosstask_loader import CrossTaskDataset
from graph_utils import dijkstra_shortest_path
from metrics import Acc_class, IoU_class
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from src.data.assembly_text import AssemblyTextDataset
from src.utils.parser import args


class Print:
    def __init__(self, hold: bool):
        self.data = []
        self.hold = hold

    def hold_and_print(self, datum):
        if not self.hold:
            print(datum)
        self.data.append(datum)

    def release(self):
        if self.hold:
            print(self.data)


print_obj = Print(args.hold_print)

clustering_distance_thresh = args.clustering_th
match_thresh = args.match_th
beam_search_thresh = args.beam_search_th


dataset = args.dataset
eval_modality = args.eval_mode
graph_type = args.graph_type
use_clusters = args.use_clusters

method = args.method
# method = 'baseline-with-cluster'

######## Parameters #####
# clustering_distance_thresh = 2.0
# match_thresh = 0.45
# beam_search_thresh = 0.48
########
keystep_thresh = 0.0
prune_keysteps = args.prune_keysteps
# not for clusters, only used when prune_keysteps is set to True
keystep_thresh = args.keysteps_th if prune_keysteps else None

data_statistics = {
    "coin": {
        "video": {"max": 24.12, "min": -23.83},
        "text": {"max": 1.0, "min": -1.0},
    },
    "crosstask": {
        "video": {"max": 27.45, "min": -18.0},
        "text": {"max": 1.0, "min": -1.0},
    },
}

if dataset == "coin":
    eval_dataset = COINDataset(modality=eval_modality, graph_type=graph_type)
elif dataset == "crosstask":
    eval_dataset = CrossTaskDataset(modality=eval_modality, graph_type=graph_type)
elif dataset == "assembly-label":
    eval_dataset = AssemblyTextDataset(path=args.dataset_path)
else:
    raise NotImplementedError

# ? these should be the speech transcripts, so I chose fine-grained annotations
text_labels = eval_dataset.get_text_labels()

# Clustering code
embedder = SentenceTransformer("all-mpnet-base-v2")

# Corpus with example sentences
corpus_embeddings = embedder.encode(text_labels)

# Normalize the embeddings to unit length
corpus_embeddings = corpus_embeddings / np.linalg.norm(
    corpus_embeddings, axis=1, keepdims=True
)

# * Perform kmean clustering
clustering_model = AgglomerativeClustering(
    n_clusters=None, distance_threshold=clustering_distance_thresh
)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(text_labels[sentence_id])

cluster_per_sentence = {}
embeds_per_cluster = {}
for key, values in clustered_sentences.items():
    # ! indices = [text_labels.index(x) for x in values]
    indices = [text_labels.index(x) for x in values]
    # indices = [np.where(np.array(text_labels) == x)[0].item() for x in values] # TODO in case you happen to have an array in AssemblyDataset, use that
    embeds_per_cluster[key] = corpus_embeddings[indices, :]
    for value in values:
        cluster_per_sentence[value] = key

# TODO missing
per_class_text = eval_dataset.text_labels_per_class
percent_match = 0.0
cluster_assignment_per_class = {}
for class_idx in per_class_text:
    chosen_clusters = []
    for text in per_class_text[class_idx]:
        chosen_clusters.append(cluster_per_sentence[text])
    mode_value = statistics.mode(chosen_clusters)
    cluster_assignment_per_class[class_idx] = mode_value
    mode_ratio = chosen_clusters.count(mode_value) / len(chosen_clusters)
    percent_match += mode_ratio
percent_match /= len(per_class_text)
print("Percent match", percent_match)
print(
    "Elements per cluster",
    sum(len(lst) for lst in clustered_sentences.values()) / len(clustered_sentences),
)
print("Num clusters: {}".format(len(clustered_sentences)))
############# Now load COIN ASR text and then see how many are correctly assigned to a cluster ###########
import json
import os
import pickle

with open("data/coin/coin.json") as f:
    coin_meta = json.load(f)["database"]

# get video classes
video_classes = []
for vid in coin_meta:
    # Obtain video classes
    if coin_meta[vid]["class"] not in video_classes:
        video_classes.append(coin_meta[vid]["class"])

# * for every video, match it to the nearest cluster
num_correct = 0
total = 0
cluster_assignment_per_video = {}
for vid in tqdm(coin_meta):
    vid_path = f"data/coin/coin_all_scores/{vid}.pkl"
    if coin_meta[vid]["subset"] == "testing" and os.path.exists(vid_path):
        curr_video_class = video_classes.index(coin_meta[vid]["class"])
        with open(vid_path, "rb") as f2:
            sentences = pickle.load(f2)
        sentences = np.concatenate(sentences)
        # now match it to per cluster embeds
        num_matches_per_cluster = {}
        for key, cluster_sents in clustered_sentences.items():
            scores = sentences[:, [text_labels.index(x) for x in cluster_sents]]
            num_matches_per_cluster[key] = (
                np.sum(scores > match_thresh) / scores.shape[1]
            )
        best_cluster = max(num_matches_per_cluster, key=num_matches_per_cluster.get)
        cluster_assignment_per_video[vid] = best_cluster
        if cluster_assignment_per_class[curr_video_class] == best_cluster:
            num_correct += 1
        total += 1
print("Total correct mapping of video to cluster: {}".format(num_correct / total))
# exit()


# * Now do baseline evaluation
result = {f"{method}": [], f"{method}-all": []}
all_preds = []
all_labels = []
if method == "baseline-with-cluster":
    for idx, datum in enumerate(tqdm(eval_dataset)):
        label = datum["labels"]
        raw_sentences = datum["sim_score"]

        preds = []
        keystep_sentences = clustered_sentences[
            cluster_assignment_per_video[datum["video_id"]]
        ]
        keystep_indices = [text_labels.index(x) for x in keystep_sentences]
        for raw_sentence in raw_sentences:
            if isinstance(raw_sentence, int) and raw_sentence == -1:
                preds.append(-1)
            else:
                curr_score = raw_sentence[:, keystep_indices]
                preds.append(keystep_indices[np.argmax(curr_score)])
        preds = np.array(preds)
        label = np.array(label)
        iou_cls = IoU_class(preds, label)
        acc_cls = Acc_class(preds, label, use_negative=False)
        result["{}".format(method)].append((acc_cls, iou_cls))
        all_preds.append(preds)
        all_labels.append(label)

########### ? WIPPPPP ##############
elif method == "beam-search-with-cluster":
    task_graph = TaskGraph(
        dataset=dataset, graph_type=graph_type, graph_modality=eval_modality
    )
    text_labels_overall = eval_dataset.get_text_labels()
    print("Populating task graph...")
    for idx, datum in enumerate(tqdm(eval_dataset)):
        label = datum["labels"]
        scores = datum["sim_score"]
        class_idx = datum["video_class"] if graph_type == "task" else -1
        if graph_type == "overall":
            text_labels = text_labels_overall
        elif graph_type == "task":
            text_labels = text_labels_overall[class_idx]
        # Populate task graph
        # Ignore -1s
        scores = [x for x in scores if not isinstance(x, int)]
        if len(scores) > 0:
            scores = np.concatenate(scores, axis=0)
            task_graph.register_sequence(
                sim=scores, keystep_sents=text_labels, class_idx=class_idx
            )
    # Normalize task graph
    task_graph.check_and_finalize(apply_log=False)
    # print(task_graph.task_graph)
    # exit()
    # CHECK: check if we can condense the candidates
    # for idx, datum in enumerate(tqdm(eval_dataset)):
    if False:
        label = datum["labels"]
        scores = datum["sim_score"]
        class_idx = datum["video_class"]
        # print(eval_dataset.text_labels_per_class[class_idx])
        # print(class_idx)
        best_collections = []
        for time_idx in range(len(scores)):
            # print(type(scores[time_idx]))
            if not isinstance(scores[time_idx], int):
                if np.max(scores[time_idx]) > 0.6:
                    if text_labels[np.argmax(scores[time_idx])] not in best_collections:
                        best_collections.append(
                            text_labels[np.argmax(scores[time_idx])]
                        )
        print("......................")
        print(eval_dataset.text_labels_per_class[class_idx])
        print(best_collections)
        # find top m next neighbors from the graph
        best_labels_count = {}
        for best_label in best_collections:
            if best_label not in best_labels_count:
                best_labels_count[best_label] = 1
            else:
                best_labels_count[best_label] += 1
            next_states = task_graph.task_graph[best_label]
            # print(next_states)
            next_best_labels = sorted(
                next_states, key=lambda x: next_states[x], reverse=True
            )[:7]
            for next_label in next_best_labels:
                if next_label != best_label:
                    if next_label not in best_labels_count:
                        best_labels_count[next_label] = 1
                    else:
                        best_labels_count[next_label] += 1
        print("After correction....")
        sorted_best_labels = {}
        for key, value in sorted(
            best_labels_count.items(), key=lambda x: x[1], reverse=True
        ):
            sorted_best_labels[key] = value
        print(sorted_best_labels)
        input()

        # if len(best_collections) > 0:
        #     print(task_graph.task_graph[best_collections[0]])
        # print('\n')
        # input()
        # print('A')
        # print(type(scores))
        # exit()
    # exit()
    # Now perform correction
    for idx, datum in enumerate(tqdm(eval_dataset)):
        label = datum["labels"]
        scores = datum["sim_score"]
        class_idx = datum["video_class"] if graph_type == "task" else -1
        if graph_type == "overall":
            text_labels = text_labels_overall
            final_task_graph = task_graph.task_graph
        elif graph_type == "task":
            text_labels = text_labels_overall[class_idx]
            final_task_graph = task_graph.task_graph[class_idx]

        if prune_keysteps and not use_clusters:
            search_nodes = []
            for score in scores:
                if not isinstance(score, int):
                    if np.max(score) > keystep_thresh:
                        if text_labels[np.argmax(score)] not in search_nodes:
                            # print(len(text_labels))
                            # print(np.argmax(scores[time_idx]))
                            search_nodes.append(text_labels[np.argmax(score)])
            # search_nodes_graph = {}
            # for node in search_nodes:
            #     if node not in search_nodes_graph:
            #         search_nodes_graph[node] = 1
            #     else:
            #         search_nodes_graph[node] += 1
            #     next_states = task_graph.task_graph[node]
            #     next_best_labels = sorted(next_states, key=lambda x: next_states[x], reverse=True)[:10]
            #     for next_label in next_best_labels:
            #         if next_label != node:
            #             if next_label not in search_nodes_graph:
            #                 search_nodes_graph[next_label] = 1
            #             else:
            #                 search_nodes_graph[next_label] += 1
            # search_nodes = sorted(search_nodes_graph, key=lambda x: search_nodes_graph[x], reverse=True)[:10]
        elif use_clusters:
            search_nodes = clustered_sentences[
                cluster_assignment_per_video[datum["video_id"]]
            ]
            # keystep_indices = [text_labels.index(x) for x in keystep_sentences]
        else:
            search_nodes = "all"
        # print(search_nodes)
        # print('Starting search nodes..............')

        preds = []
        # Replace the low scores with -1
        for score in scores:
            if isinstance(score, int) and score == -1:
                preds.append(-1)
            else:
                # Normalize score to be in the range [-1, 1]
                score = (
                    2
                    * (score - data_statistics[dataset][eval_modality]["min"])
                    / (
                        data_statistics[dataset][eval_modality]["max"]
                        - data_statistics[dataset][eval_modality]["min"]
                    )
                    - 1
                )
                if search_nodes == "all" or len(search_nodes) == 0:
                    if np.max(score) < beam_search_thresh:
                        preds.append(-1)
                    else:
                        preds.append(int(np.argmax(score)))
                else:
                    focus_labels = [text_labels.index(x) for x in search_nodes]
                    subset_score = score[:, focus_labels]
                    max_val = np.max(subset_score)
                    max_idx = np.argmax(subset_score)
                    orig_max_idx = focus_labels[max_idx]
                    if max_val < beam_search_thresh:
                        preds.append(-1)
                    else:
                        preds.append(orig_max_idx)
        # Replace -1s with beam search
        # print('Before........')
        # for p_idx, p_pred in enumerate(preds):
        #     print(p_idx, p_pred)
        preds_before = preds.copy()
        prev_label_time = -1
        prev_label = -1
        for time_idx in range(len(preds)):
            if preds[time_idx] != -1:
                # Do backward correction
                curr_label_time = time_idx
                curr_label = preds[time_idx]

                if curr_label_time - prev_label_time == 1:
                    continue

                if prev_label != -1:
                    best_path, weight = dijkstra_shortest_path(
                        final_task_graph,
                        text_labels[prev_label],
                        text_labels[curr_label],
                        search_nodes=search_nodes,
                    )
                    # print(text_labels[prev_label], text_labels[curr_label])
                    # print(search_nodes)
                    # print(best_path)
                    # exit()
                else:
                    best_path = [text_labels[curr_label]]
                    prev_label = curr_label

                # We need to update starting prev_label_time+1 to curr_label_time
                if len(best_path) > 0:
                    update_labels = (
                        [prev_label]
                        + [text_labels.index(y) for y in best_path]
                        + [curr_label]
                    )
                else:
                    update_labels = [prev_label, curr_label]
                n_segs = len(update_labels)
                for count_idx, corrected_time_idx in enumerate(
                    range(prev_label_time + 1, curr_label_time)
                ):
                    preds[corrected_time_idx] = update_labels[
                        int(
                            n_segs
                            * count_idx
                            / len(range(prev_label_time + 1, curr_label_time))
                        )
                    ]

                prev_label_time = curr_label_time
                prev_label = curr_label
        # print('After........')
        # for p_idx, p_pred in enumerate(preds):
        #     print(p_idx, p_pred, preds_before[p_idx])
        # exit()
        preds = np.array(preds)
        label = np.array(label)
        iou_cls = IoU_class(preds, label)
        acc_cls = Acc_class(preds, label, use_negative=False)
        result["{}".format(method)].append((acc_cls, iou_cls))
        all_preds.append(preds)
        all_labels.append(label)

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
result["{}-all".format(method)].append(
    (
        Acc_class(all_preds, all_labels, use_negative=False),
        IoU_class(all_preds, all_labels),
    )
)

for name in result:
    acc = np.mean([r[0] for r in result[name]])
    iou = np.mean([r[1] for r in result[name]])
    print(
        f"Clustering thresh: {clustering_distance_thresh}, Match thresh: {match_thresh}, beam_search_thresh: {beam_search_thresh}, {name} | Acc {acc}, IoU: {iou}"
    )

exit()

clustered_items = 0
for i, cluster in clustered_sentences.items():
    print("Cluster ", i + 1)
    print(cluster)
    clustered_items += len(cluster)
    print("")

print(clustered_items)
