import sys
import numpy as np
import lmdb
import pyarrow as pa
import pickle
import networkx as nx
from networkx.algorithms.dag import lexicographical_topological_sort

from os import path as osp
from glob import glob
from torch.utils.data import Dataset

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))  # add parent dir
from data.data_utils import Time2FrameNumber
from dp.graph_utils import remove_nodes_from_graph, generate_metagraph


class LMDB_Folder_Dataset(Dataset):
    def __init__(
        self,
        folder,
        split="train",
        transform=None,
        activity_type="all",
        map_node_features=False,
        features="mil",
        graph_type="manual",
    ):
        # filtering out folders that have desired split
        self.framerate = 32 if features == "mil" else 10
        self.map_node_features = map_node_features
        # self.map_node_features = True
        self.transform = transform
        cls_folders = []
        for cls_folder in glob(osp.join(folder, "*/")):
            files = glob(osp.join(cls_folder, "*.lmdb"))
            file_has_split = ["_{}".format(split) in f for f in files]
            if any(file_has_split):
                cls_folders.append(cls_folder)

        if activity_type != "all":
            # loading information about activities to be able to filter them
            steps_info_filename = osp.join("/".join(folder.split("/")[:-1]), "steps_info.pickle")
            with open(steps_info_filename, "rb") as handle:
                steps_info = pickle.load(handle)
            # filtering the activities by type
            cls_folders = [f for f in cls_folders if steps_info["cls_to_type"][int(f.split("_")[-2])] == activity_type]

        # instantiating datasets for each class
        self.cls_datasets = [
            LMDB_Class_Dataset(f, split, transform, 0, framerate=self.framerate) for f in sorted(cls_folders)
        ]
        self.cls_lens = [len(d) for d in self.cls_datasets]
        self.cls_end_idx = np.cumsum(self.cls_lens)

        root = "/".join(folder.split("/")[:-1])
        if graph_type == "manual":
            graph_info_filename = osp.join(root, "graphs", "crosstask_recipes_manual_full.pickle")
        elif graph_type == "dhaivat":
            graph_info_filename = osp.join(root, "graphs", "dhaivat_graph.pickle")
        elif graph_type == "hai":
            graph_info_filename = osp.join(root, "graphs", "hai_graph.pickle")
        self.graph_loader = GraphLoader(graph_info_filename)

    def get_step_embedding(self, step_idx):
        for cls_dataset in self.cls_datasets:
            if step_idx in cls_dataset.step_embeddings:
                return cls_dataset.step_embeddings.get(step_idx)

    def get_step_description(self, step_idx):
        for cls_dataset in self.cls_datasets:
            if step_idx in cls_dataset.step_descriptions:
                return cls_dataset.step_descriptions.get(step_idx)

    def __getitem__(self, idx):
        # find which dataset the idx corresponds to
        dataset_idx = 0
        while idx >= self.cls_end_idx[dataset_idx]:
            dataset_idx += 1

        # find the relative idx within selected dataset
        start_idx = 0 if dataset_idx == 0 else self.cls_end_idx[dataset_idx - 1]
        relative_idx = idx - start_idx
        cls_dataset = self.cls_datasets[dataset_idx]
        cls_name = cls_dataset.cls_id
        sample = cls_dataset[relative_idx]
        sample["node_features"] = self.graph_loader.get_class_features(cls_name)
        sample["node_ids"] = self.graph_loader.get_class_nodes(cls_name)
        sample["metagraph"] = self.graph_loader.get_class_metagraph(cls_name)
        sample["node2step"] = self.graph_loader.get_class_local_r2c(cls_name)
        if self.map_node_features:
            take_step_idxs = [s for n, s in sample["node2step"].items() if s > -1]
            unique_step_idxs = []
            for s in take_step_idxs:
                if s not in unique_step_idxs:
                    unique_step_idxs.append(s)
            sample["node_features"] = sample["all_step_features"][unique_step_idxs]
            sample["node_ids"] = np.arange(len(unique_step_idxs))
            # for node_id in range(sample["node_features"].shape[0]):
            #     step_id = node2step[node_id]
            #     if step_id > -1:
            #         sample["node_features"][node_id] = sample["all_step_features"][step_id]
        return self.transform(sample)

    def __len__(self):
        return sum(self.cls_lens)

    def load_graph(self):
        graph_path = os.path.join(CT_PATH, "graphs", "crosstask_recipes_manual_full.json")
        with open(graph_path, "r") as f:
            graph_annots = json.load(f)
            graph_annots = {s["recipe_type"]: s for s in graph_annots}


class LMDB_Class_Dataset(Dataset):
    def __init__(self, cls_folder, split="train", transform=None, truncate=0, framerate=32):
        self.framerate = framerate
        split_name = osp.basename(cls_folder.rstrip("/")).split("_")
        if len(split_name) == 2:
            self.cls_id = int(split_name[0])
        else:
            self.cls_id = int(split_name[1])
        # self.cls_id = int(osp.basename(cls_folder.rstrip('/')).split('_')[1])
        lmdb_filename = osp.basename(cls_folder.rstrip("/")).replace("lmdb", split) + ".lmdb"
        db_path = osp.join(cls_folder, lmdb_filename)
        self.env = lmdb.open(
            db_path, subdir=osp.isdir(db_path), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b"__len__"))
            self.keys = sorted(pa.deserialize(txn.get(b"__keys__")))

        if truncate > 0:
            self.length = min(truncate, self.length)

        self.transform = transform

        # reading step embeddings
        dataset_root_dir = "/".join(cls_folder.split("/")[:-3])
        steps_info_filename = osp.join(dataset_root_dir, "steps_info.pickle")
        if osp.exists(steps_info_filename):
            with open(steps_info_filename, "rb") as handle:
                steps_info = pickle.load(handle)
            # take only the embeddings that belong to this class to save memory
            self.step_embeddings = {
                k: v
                for k, v in steps_info["steps_to_embeddings"].items()
                if k in steps_info["cls_to_steps"][self.cls_id]
            }
            self.step_descriptions = {
                k: v
                for k, v in steps_info["steps_to_descriptions"].items()
                if k in steps_info["cls_to_steps"][self.cls_id]
            }
        else:
            self.step_embeddings = {}
            self.step_descriptions = {}

    def __getitem__(self, idx):
        """
        Extracts a sample with the following fields:

        Attributes of sample_dict
        ----------
        name: name of the example
        cls: class (or activity) of the video
        cls_name: name of the class (or activity)
        num_frames: Number of frames N
        frame_features: features from all N frames in the video, has size [N, d]
        num_steps: Number of steps K
        step_ids: ids of steps that happen in this video, has len K
        step_features: Feature matrix of size [K, d], where d
        step_starts: start of each step (in frames)
        step_ends: end of each step (in frames)
        """
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx])
        full_sample_dict = pa.deserialize(byteflow)
        sample_dict = {
            k: v for k, v in full_sample_dict.items() if k in ["name", "cls", "cls_name", "num_steps", "num_subs"]
        }
        sample_dict["frame_features"] = full_sample_dict["frames_features"]
        sample_dict["num_frames"] = np.array(full_sample_dict["frames_features"].shape[0])

        # fill in the dict with step features and their durations
        sample_dict["step_ids"] = full_sample_dict["steps_ids"]
        sample_dict["all_step_ids"] = np.array(list(self.step_descriptions.keys()))
        # get step features
        if "steps_features" in full_sample_dict.keys():
            sample_dict["step_features"] = full_sample_dict["steps_features"]
            sample_dict["all_step_features"] = sample_dict["step_features"]
        else:
            sample_dict["step_features"] = np.concatenate([self.step_embeddings[k] for k in sample_dict["step_ids"]])
            sample_dict["all_step_features"] = np.concatenate(
                [self.step_embeddings[k] for k in sample_dict["all_step_ids"]]
            )
        # transform seconds to steps
        sample_dict["step_starts_sec"] = full_sample_dict["steps_starts"]
        sample_dict["step_starts"] = np.array(
            [Time2FrameNumber(s, 10) // self.framerate for s in full_sample_dict["steps_starts"]]
        )

        sample_dict["step_ends_sec"] = full_sample_dict["steps_ends"]
        sample_dict["step_ends"] = np.array(
            [Time2FrameNumber(s, 10) // self.framerate for s in full_sample_dict["steps_ends"]]
        )

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        return sample_dict

    def __len__(self):
        return self.length

    # def __repr__(self):
    #     return self.__class__.__name__ + " (" + self.db_path + ")"


class GraphLoader:
    def __init__(self, graph_info_filename, remove_nodes=False):
        self.remove_nodes = remove_nodes
        print(graph_info_filename)
        with open(graph_info_filename, "rb") as handle:
            self.graph_info = pickle.load(handle)

        for name, info_dict in self.graph_info.items():
            local_r2c = info_dict["local_recipe_2_caption"]
            if remove_nodes:
                info_dict["present_nodes"] = [r for r, c in local_r2c.items() if c > -1]
                info_dict["absent_nodes"] = [r for r, c in local_r2c.items() if c == -1]
            else:
                info_dict["present_nodes"] = np.arange(len(info_dict["instructions"]))
                info_dict["absent_nodes"] = []

        self.build_graphs()

    def build_graphs(self):
        for name, info_dict in self.graph_info.items():
            local_edges, instructions = info_dict["local_edges"], info_dict["instructions"]

            G = nx.DiGraph()
            [G.add_node(i) for i in range(len(instructions))]
            [G.add_edge(n1, n2) for n1, n2 in local_edges]

            if self.remove_nodes:
                G = remove_nodes_from_graph(G, absent_nodes, relabel=False)
                nodes, edges = list(G.nodes()), list(G.edges())

                all_to_present = {p: i for i, p in enumerate(graph_info["present_nodes"])}
                G = nx.DiGraph()
                [G.add_node(all_to_present[i]) for i in nodes]
                [G.add_edge(all_to_present[i], all_to_present[j]) for i, j in edges]

            info_dict["graph"] = G
            info_dict["metagraph"] = generate_metagraph(G)

    def get_class_features(self, cls_name):
        dic = self.graph_info[cls_name]
        return dic["instruction_features"][dic["present_nodes"]]

    def get_class_nodes(self, cls_name):
        dic = self.graph_info[cls_name]
        global_node_ids = np.array([r for r, c in dic["global_recipe_2_caption"]])
        return global_node_ids[dic["present_nodes"]]

    def get_class_metagraph(self, cls_name):
        dic = self.graph_info[cls_name]
        return dic["metagraph"]

    def get_class_local_r2c(self, cls_name):
        dic = self.graph_info[cls_name]
        return dict(dic["local_recipe_2_caption"])
