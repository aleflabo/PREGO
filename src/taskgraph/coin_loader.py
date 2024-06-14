import json
import math
import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from tqdm import tqdm


class COINDataset(Dataset):
    def __init__(
        self,
        modality="text",
        graph_type="overall",
        use_cache=False,
        return_raw_sentences=False,
    ):
        self.modality = modality
        self.graph_type = graph_type
        self.return_raw_sentences = return_raw_sentences
        # TODO for now we assume all these things do not exist, try with them
        root_path = "data/coin"
        # root_path = ""
        COIN_ANNOTATIONS_PATH = os.path.join("data", "coin", "coin.json")
        # COIN_ANNOTATIONS_PATH = os.path.join(root_path, "coin.json")
        self.SENTENCES_SAVE_PATH = os.path.join(root_path, "coin_processed")
        self.SCORES_PATH = os.path.join(root_path, "coin_all_scores")
        # self.COIN_VIDEO_FEATS_PATH = '/checkpoint/ashutoshkr/videoclip/coin_feats/'
        self.COIN_VIDEO_FEATS_PATH = (
            "/checkpoint/ashutoshkr/iv_nlq_s23/videoclip_video_features/"
        )
        with open(COIN_ANNOTATIONS_PATH) as f:
            self.coin_meta = json.load(f)["database"]

        self.coin_text_labels = []
        self.video_classes = []
        coin_durations = []
        for vid in tqdm(self.coin_meta):
            # Obtain text labels
            for annotation in self.coin_meta[vid]["annotation"]:
                if annotation["label"] not in self.coin_text_labels:
                    self.coin_text_labels.append(annotation["label"])
            # Obtain durations
            coin_durations.append(self.coin_meta[vid]["duration"])
            # Obtain video classes
            if self.coin_meta[vid]["class"] not in self.video_classes:
                self.video_classes.append(self.coin_meta[vid]["class"])
        # To answer reviewer's question, we introduce some noise from HowTo100M keysteps
        import random

        # random.seed(0)
        # coin_orig = 749
        # multiplier = 5
        # if multiplier >= 1.0:
        #     with open(
        #         "/private/home/ashutoshkr/code/iv_nlq_s23/video-distant-supervision/data/step_label_text.json"
        #     ) as fh:
        #         ht100m_keysteps_data = json.load(fh)
        #     ht100m_keysteps = []
        #     for i in range(len(ht100m_keysteps_data)):
        #         for j in range(len(ht100m_keysteps_data[i])):
        #             ht100m_keysteps.append(
        #                 ht100m_keysteps_data[i][j]["headline"].strip()
        #             )
        #     ht100m_keysteps_subset = random.sample(
        #         ht100m_keysteps, int(coin_orig * (multiplier - 1.0))
        #     )
        #     self.coin_text_labels.extend(ht100m_keysteps_subset)
        # else:
        #     self.coin_text_labels = random.sample(
        #         self.coin_text_labels, int(coin_orig * multiplier)
        #     )

        self.text_labels_per_class = {}
        for vid in tqdm(self.coin_meta):
            # Obtain text labels per class
            if (
                self.video_classes.index(self.coin_meta[vid]["class"])
                not in self.text_labels_per_class
            ):
                self.text_labels_per_class[
                    self.video_classes.index(self.coin_meta[vid]["class"])
                ] = []
            for annotation in self.coin_meta[vid]["annotation"]:
                if (
                    annotation["label"]
                    not in self.text_labels_per_class[
                        self.video_classes.index(self.coin_meta[vid]["class"])
                    ]
                ):
                    self.text_labels_per_class[
                        self.video_classes.index(self.coin_meta[vid]["class"])
                    ].append(annotation["label"])

        # # To answer reviewer's question, we introduce some noise from non-task categories
        # coin_orig = 749
        # multiplier = 1.
        # # print(len(self.text_labels_per_class))
        # for given_class in self.text_labels_per_class:
        #     not_used_keysteps = [x for x in self.coin_text_labels if x not in self.text_labels_per_class[given_class]]
        #     num_samples = int(coin_orig * multiplier - len(self.text_labels_per_class[given_class]))
        #     self.text_labels_per_class[given_class].extend(random.sample(not_used_keysteps, num_samples))
        #     # print(len(self.text_labels_per_class[given_class]))
        # # exit()

        if self.modality == "text":
            self.data = [
                (x, coin_durations[idx])
                for idx, x in enumerate(self.coin_meta)
                if os.path.exists(os.path.join(self.SENTENCES_SAVE_PATH, f"{x}.json"))
                and self.coin_meta[x]["subset"] == "testing"
            ]
        elif self.modality == "video":
            self.data = [
                (x, coin_durations[idx])
                for idx, x in enumerate(self.coin_meta)
                if os.path.exists(
                    os.path.join(self.COIN_VIDEO_FEATS_PATH, "{}.pt".format(x))
                )
                and self.coin_meta[x]["subset"] == "testing"
            ]
        elif self.modality == "video-text":
            self.data = [
                (x, coin_durations[idx])
                for idx, x in enumerate(self.coin_meta)
                if os.path.exists(
                    os.path.join(self.COIN_VIDEO_FEATS_PATH, "{}.pt".format(x))
                )
                and os.path.exists(os.path.join(self.SENTENCES_SAVE_PATH, f"{x}.json"))
                and self.coin_meta[x]["subset"] == "testing"
            ]
        else:
            ValueError, "Unknown modality"
        self.use_cache = use_cache

        self.model = SentenceTransformer("all-mpnet-base-v2")
        if not use_cache:
            self.model = SentenceTransformer("all-mpnet-base-v2")
            self.label_embeds = self.model.encode(self.coin_text_labels)
            # print(label_embeds.shape)
            # exit()
        print("Dataset length: {}".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def get_text_labels(self):
        if self.graph_type == "overall":
            return self.coin_text_labels
        elif self.graph_type == "task":
            return self.text_labels_per_class

    def __getitem__(self, index):
        vidid, duration = self.data[index]
        ####### Generate labels ###########
        labels = np.full(int(duration), -1)
        curr_video_class = self.video_classes.index(self.coin_meta[vidid]["class"])
        for caption_idx in range(len(self.coin_meta[vidid]["annotation"])):
            segment = self.coin_meta[vidid]["annotation"][caption_idx]["segment"]
            text_label = self.coin_meta[vidid]["annotation"][caption_idx]["label"]
            for time_idx in range(math.floor(segment[0]), math.ceil(segment[1])):
                if time_idx < len(labels):
                    labels[time_idx] = self.coin_text_labels.index(text_label)
                    assert text_label in self.text_labels_per_class[curr_video_class]
        ###################################
        ############## Loading text scores ###########
        if self.modality == "text" or self.modality == "video-text":
            with open(os.path.join(self.SENTENCES_SAVE_PATH, f"{vidid}.json")) as f:
                captions = json.load(f)
            if self.use_cache:
                with open(
                    os.path.join(self.SCORES_PATH, "{}.pkl".format(vidid)), "rb"
                ) as fs:
                    text_scores = pickle.load(fs)
            else:
                # text_scores = []
                all_sent_embed = self.model.encode(captions["text"])
                sim_scores = np.matmul(all_sent_embed, self.label_embeds.transpose())
                text_scores = [
                    sim_scores[i, :].reshape(1, -1) for i in range(sim_scores.shape[0])
                ]
                # for sent in captions['text']:
                #     sent_embed = self.model.encode([sent])
                #     # print(sent_embed.shape)
                #     sim_scores = np.matmul(sent_embed, self.label_embeds.transpose())
                #     text_scores.append(sim_scores)

            assert len(text_scores) == len(captions["text"])
            # pass the text scores as a list and insert -1 where ASR is missing
            video_scores = [-1] * len(labels)
            if self.return_raw_sentences:
                self.raw_sentences = [-1] * len(labels)
            for idx, (start, end, text) in enumerate(
                zip(captions["start"], captions["end"], captions["text"])
            ):
                if self.return_raw_sentences:
                    text_embd = self.model.encode([text])
                for time_idx in range(math.floor(start), math.ceil(end)):
                    if time_idx < len(labels):
                        video_scores[time_idx] = text_scores[idx]
                        if self.return_raw_sentences:
                            self.raw_sentences[time_idx] = text_embd
            if self.modality == "video-text":
                text_features = video_scores
        ##############################################
        ############# Loading video scores ###########
        if self.modality == "video" or self.modality == "video-text":
            video_feats = torch.load(
                os.path.join(self.COIN_VIDEO_FEATS_PATH, "{}.pt".format(vidid))
            )
            video_scores = [-1] * len(labels)
            for i in range(len(video_feats)):
                if i < len(labels):
                    video_scores[i] = np.expand_dims(video_feats[i].numpy(), axis=0)
            if self.modality == "video-text":
                visual_features = video_scores
        ##############################################
        ############# Correct labels and sim_scores if graph_type == 'task' #########
        if self.graph_type == "task":
            if self.modality == "video-text":
                raise NotImplementedError
            # Given texts, find indices of them
            task_text_index = [
                self.coin_text_labels.index(x)
                for x in self.text_labels_per_class[curr_video_class]
            ]

            for time_idx in range(len(labels)):
                if labels[time_idx] != -1:
                    assert labels[time_idx] in task_text_index
                    labels[time_idx] = task_text_index.index(labels[time_idx])

            for i in range(len(video_scores)):
                if not isinstance(video_scores[i], int):
                    video_scores[i] = video_scores[i][:, task_text_index]
        #############################################################################
        if self.return_raw_sentences:
            return {
                "labels": labels,
                "sim_score": {"video": visual_features, "text": text_features}
                if self.modality == "video-text"
                else video_scores,
                "video_id": self.data[index][0],
                "video_class": curr_video_class,
                "raw_sentences": self.raw_sentences,
            }
        return {
            "labels": labels,
            "sim_score": {"video": visual_features, "text": text_features}
            if self.modality == "video-text"
            else video_scores,
            "video_id": self.data[index][0],
            "video_class": curr_video_class,
        }


if __name__ == "__main__":
    dataset = COINDataset(modality="video")
    # print(dataset[0])
    for i in range(10):
        dataset[i]
    print(len(dataset))
