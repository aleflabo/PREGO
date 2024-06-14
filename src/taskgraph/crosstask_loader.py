import json
import math
import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from tqdm import tqdm


class CrossTaskDataset(Dataset):
    def __init__(self, modality="text", graph_type="overall", use_cache=False):
        self.modality = modality
        self.graph_type = graph_type
        self.CROSSTASK_ANNOTATIONS_PATH = (
            "/datasets01/CrossTask/053122/crosstask_release/annotations/"
        )
        self.SENTENCES_SAVE_PATH = "/private/home/ashutoshkr/code/iv_nlq_s23/graph_check_crosstask_coin/crosstask_processed/"
        self.SCORES_PATH = "/private/home/ashutoshkr/code/iv_nlq_s23/graph_check_crosstask_coin/crosstask_all_scores/"
        self.CROSSTASK_VIDEO_FEATS_PATH = (
            "/checkpoint/ashutoshkr/iv_nlq_s23/videoclip_video_features_crosstask_s3d/"
        )
        crosstask_videos = [
            (int(x[:-16]), x[-15:-4])
            for x in os.listdir(self.CROSSTASK_ANNOTATIONS_PATH)
        ]

        # get crosstask_labels
        self.crosstask_text_labels = []
        self.text_labels_per_class = {}
        self.video_classes = []
        self.non_collapsed_labels = (
            {}
        )  # useful since crosstask per video annotation has label index only
        with open(
            "/datasets01/CrossTask/053122/crosstask_release/tasks_primary.txt"
        ) as f:
            lines = f.readlines()
            tasks = [x.strip() for idx, x in enumerate(lines) if idx % 6 == 4]
            self.video_classes = [
                int(x.strip()) for idx, x in enumerate(lines) if idx % 6 == 0
            ]
        for task, class_idx in zip(tasks, self.video_classes):
            # print(class_idx, task)
            steps = task.split(",")
            self.non_collapsed_labels[class_idx] = steps
            for step in steps:
                if step not in self.crosstask_text_labels:
                    self.crosstask_text_labels.append(step)
                # add to per task classes
                if (
                    self.video_classes.index(int(class_idx))
                    not in self.text_labels_per_class
                ):
                    self.text_labels_per_class[
                        self.video_classes.index(int(class_idx))
                    ] = []
                if (
                    step
                    not in self.text_labels_per_class[
                        self.video_classes.index(int(class_idx))
                    ]
                ):
                    self.text_labels_per_class[
                        self.video_classes.index(int(class_idx))
                    ].append(step)

        # To answer reviewer's question, we introduce some noise from HowTo100M keysteps
        import random

        random.seed(0)
        coin_orig = 105
        multiplier = 10.0
        if multiplier >= 1.0:
            with open(
                "/private/home/ashutoshkr/code/iv_nlq_s23/video-distant-supervision/data/step_label_text.json"
            ) as fh:
                ht100m_keysteps_data = json.load(fh)
            ht100m_keysteps = []
            for i in range(len(ht100m_keysteps_data)):
                for j in range(len(ht100m_keysteps_data[i])):
                    ht100m_keysteps.append(
                        ht100m_keysteps_data[i][j]["headline"].strip()
                    )
            ht100m_keysteps_subset = random.sample(
                ht100m_keysteps, int(coin_orig * (multiplier - 1.0))
            )
            self.crosstask_text_labels.extend(ht100m_keysteps_subset)
        else:
            self.crosstask_text_labels = random.sample(
                self.crosstask_text_labels, int(coin_orig * multiplier)
            )

        if self.modality == "text":
            self.data = [
                x
                for x in crosstask_videos
                if os.path.exists(
                    os.path.join(self.SENTENCES_SAVE_PATH, "{}.json".format(x[1]))
                )
            ]
        elif self.modality == "video":
            self.data = [
                x
                for x in crosstask_videos
                if os.path.exists(
                    os.path.join(self.CROSSTASK_VIDEO_FEATS_PATH, "{}.pt".format(x[1]))
                )
            ]
            self.data = [
                x for x in self.data if "bo355kAfADM" not in x[1]
            ]  # incomplete video
        elif self.modality == "video-text":
            self.data = [
                x
                for x in crosstask_videos
                if os.path.exists(
                    os.path.join(self.SENTENCES_SAVE_PATH, "{}.json".format(x[1]))
                )
                and os.path.exists(
                    os.path.join(self.CROSSTASK_VIDEO_FEATS_PATH, "{}.pt".format(x[1]))
                )
            ]
            self.data = [
                x for x in self.data if "bo355kAfADM" not in x[1]
            ]  # incomplete video
        else:
            ValueError, "Unknown modality"
        self.use_cache = use_cache

        if not use_cache:
            self.model = SentenceTransformer("all-mpnet-base-v2")
            self.label_embeds = self.model.encode(self.crosstask_text_labels)
            print(self.label_embeds.shape)
        print("Dataset length: {}".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def get_text_labels(self):
        if self.graph_type == "overall":
            return self.crosstask_text_labels
        elif self.graph_type == "task":
            return self.text_labels_per_class

    def __getitem__(self, index):
        curr_video_class, vidid = self.data[index]
        ####### Get the duration #########
        # Let us choose duration as the max of video feature length or annotation, fairly accurate
        duration = -1
        if os.path.exists(
            os.path.join(self.CROSSTASK_VIDEO_FEATS_PATH, "{}.pt".format(vidid))
        ):
            video_feats = torch.load(
                os.path.join(self.CROSSTASK_VIDEO_FEATS_PATH, "{}.pt".format(vidid))
            )
            duration = video_feats.shape[0]
        with open(
            os.path.join(
                self.CROSSTASK_ANNOTATIONS_PATH,
                "{}_{}.csv".format(curr_video_class, vidid),
            )
        ) as f:
            end_times = [float(x.strip().split(",")[-1]) for x in f.readlines()]
        duration = max(duration, max(end_times))
        ##################################
        ####### Generate labels ###########
        labels = np.full(int(duration), -1)
        with open(
            os.path.join(
                self.CROSSTASK_ANNOTATIONS_PATH,
                "{}_{}.csv".format(curr_video_class, vidid),
            )
        ) as f:
            annotations = [
                {
                    "mapped_label": self.crosstask_text_labels.index(
                        self.non_collapsed_labels[curr_video_class][
                            int(x.split(",")[0]) - 1
                        ]
                    ),
                    "label_text": self.non_collapsed_labels[curr_video_class][
                        int(x.split(",")[0]) - 1
                    ],
                    "start_time": float(x.split(",")[1]),
                    "end_time": float(x.strip().split(",")[2]),
                }
                for x in f.readlines()
            ]
        for caption_idx in range(len(annotations)):
            text_label = annotations[caption_idx]["label_text"]
            for time_idx in range(
                math.floor(annotations[caption_idx]["start_time"]),
                math.ceil(annotations[caption_idx]["end_time"]),
            ):
                if time_idx < len(labels):
                    labels[time_idx] = annotations[caption_idx]["mapped_label"]
                    assert (
                        text_label
                        in self.text_labels_per_class[
                            self.video_classes.index(curr_video_class)
                        ]
                    )
        ###################################
        ############## Loading text scores ###########
        if self.modality == "text" or self.modality == "video-text":
            with open(
                os.path.join(self.SENTENCES_SAVE_PATH, "{}.json".format(vidid))
            ) as f:
                captions = json.load(f)
            if self.use_cache:
                with open(
                    os.path.join(self.SCORES_PATH, "{}.pkl".format(vidid)), "rb"
                ) as fs:
                    text_scores = pickle.load(fs)
            else:
                all_sent_embed = self.model.encode(captions["text"])
                sim_scores = np.matmul(all_sent_embed, self.label_embeds.transpose())
                text_scores = [
                    sim_scores[i, :].reshape(1, -1) for i in range(sim_scores.shape[0])
                ]
            assert len(text_scores) == len(captions["text"])
            # pass the text scores as a list and insert -1 where ASR is missing
            video_scores = [-1] * len(labels)
            for idx, (start, end) in enumerate(zip(captions["start"], captions["end"])):
                for time_idx in range(math.floor(start), math.ceil(end)):
                    if time_idx < len(labels):
                        video_scores[time_idx] = text_scores[idx]
            if self.modality == "video-text":
                text_features = video_scores
        ##############################################
        ############# Loading video scores ###########
        if self.modality == "video" or self.modality == "video-text":
            video_feats = torch.load(
                os.path.join(self.CROSSTASK_VIDEO_FEATS_PATH, "{}.pt".format(vidid))
            )
            video_scores = [-1] * len(labels)
            if video_feats.shape[0] < len(labels):
                print("DDD: ", video_feats.shape[0], len(labels), vidid)
            for i in range(len(video_feats)):
                if i < len(labels):
                    video_scores[i] = np.expand_dims(video_feats[i].numpy(), axis=0)
            if self.modality == "video-text":
                visual_features = video_scores
        ##############################################
        ############# Correct labels and sim_scores if graph_type == 'task' #########
        if self.graph_type == "task":
            if self.modality == "video-task":
                raise NotImplementedError
            # Given texts, find indices of them
            task_text_index = [
                self.crosstask_text_labels.index(x)
                for x in self.text_labels_per_class[
                    self.video_classes.index(curr_video_class)
                ]
            ]

            for time_idx in range(len(labels)):
                if labels[time_idx] != -1:
                    assert labels[time_idx] in task_text_index
                    labels[time_idx] = task_text_index.index(labels[time_idx])

            for i in range(len(video_scores)):
                if not isinstance(video_scores[i], int):
                    video_scores[i] = video_scores[i][:, task_text_index]
        #############################################################################
        return {
            "labels": labels,
            "sim_score": {"video": visual_features, "text": text_features}
            if self.modality == "video-text"
            else video_scores,
            "video_id": vidid,  # self.data[index][0],
            "video_class": curr_video_class,
        }


if __name__ == "__main__":
    dataset = CrossTaskDataset(modality="video")
    # print(dataset[0])
    for i in range(10):
        dataset[i]
    print(len(dataset))
