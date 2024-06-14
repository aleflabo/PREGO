import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.loader_graph import LMDB_Folder_Dataset
from data.batching import BatchIdxSampler_Class, flatten_batch, dumb_batch
from data.data_utils import dict2tensor
from paths import CT_PATH, COIN_PATH, YC_PATH


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, n_cls, batch_size, features="mil", graph_type="manual"):
        super().__init__()
        activity = "all"
        if dataset_name == "COIN":
            folder = COIN_PATH
        elif dataset_name == "YouCook2":
            folder = YC_PATH
        elif dataset_name == "CrossTask":
            folder = CT_PATH
            activity = "primary"
        else:
            raise f"No such dataset {dataset_name}"

        self.lmdb_path = os.path.join(folder, "lmdb")
        self.n_cls = n_cls
        self.batch_size = batch_size
        transform = dict2tensor

        self.train_dataset = LMDB_Folder_Dataset(
            self.lmdb_path,
            split="train",
            transform=transform,
            activity_type=activity,
            features=features,
            graph_type=graph_type,
        )
        self.val_dataset = LMDB_Folder_Dataset(
            self.lmdb_path,
            split="val",
            transform=transform,
            activity_type=activity,
            features=features,
            graph_type=graph_type,
        )
        self.test_dataset = LMDB_Folder_Dataset(
            self.lmdb_path,
            split="test",
            transform=transform,
            activity_type=activity,
            features=features,
            graph_type=graph_type,
        )
        print(len(self.train_dataset), len(self.val_dataset))

    def train_dataloader(self):
        batch_idx_sampler = BatchIdxSampler_Class(self.train_dataset, self.n_cls, self.batch_size)
        train_loader = DataLoader(
            self.train_dataset, batch_sampler=batch_idx_sampler, collate_fn=dumb_batch, num_workers=2
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, collate_fn=dumb_batch, batch_size=self.batch_size, num_workers=2)
        return val_loader

    def test_dataloader(self):
        val_loader = DataLoader(self.test_dataset, collate_fn=dumb_batch, num_workers=2)
        return val_loader
