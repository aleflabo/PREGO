from torch.utils.data import DataLoader, Dataset

from src.data.dataset_utils import get_OH_data


class AssemblyLabelDataset(Dataset):
    def __init__(self, csv_path, split="correct"):
        assert split in [
            "correct",
            "mistake",
            "all",
        ], "split must be 'correct', 'mistake' or 'all', not '{}'".format(split)
        (
            self.oh_samplelist,
            self.oh_labellist,
            self.metadata,
            self.all_keysteps,
        ) = get_OH_data(csv_path, split)

    def __len__(self):
        return len(self.oh_samplelist)

    def __getitem__(self, idx):
        sample = {
            "oh_sample": self.oh_samplelist[idx],
            "oh_label": self.oh_labellist[idx],
            "metadata": self.metadata[idx],
        }

        return sample


def collate_fn(batch):
    print("collate_fn")

    return batch


if __name__ == "__main__":
    path_to_csv_A101 = "./mistake_labels/"

    # train_dataset = AssemblyLabelDataset(path_to_csv_A101, is_train=True)
    # test_dataset = AssemblyLabelDataset(path_to_csv_A101, is_train=False)

    train_dataset = AssemblyLabelDataset(path_to_csv_A101, split="all")
    test_dataset = AssemblyLabelDataset(path_to_csv_A101, split="mistake")
    batch_size = 32  # Adjust batch size as needed
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    for batch in train_dataloader:
        print(batch)
        break

    print(len(train_dataset))
    print(len(test_dataset))
