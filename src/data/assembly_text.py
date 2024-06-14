import re
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTModel,
    OpenAIGPTTokenizer,
    pipeline,
    set_seed,
)

from src.utils.variables import CORRECT, WRONG


class AssemblyTextDataset(Dataset):
    def __init__(self, path: str, split: str = "train") -> None:
        self.dir = Path(path)
        self.word2idx = {}
        self.idx2word = [
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[PAD]",
            "[UNK]",
        ]

        if split == "train":
            files = CORRECT
        elif split == "test":
            files = WRONG
        else:
            raise ValueError("split should be either train or test")

        self.files = [self.dir / f for f in files]
        self.__init_data__()

    def __init_data__(self) -> None:
        self.data = {}
        for f in self.files:
            df = pd.read_csv(f)
            self.data[f.stem] = (
                df[["verb", "this", "that"]]
                .apply(
                    # ! Changed " " to "-"
                    lambda x: "-".join(map(lambda y: y.replace(" ", ""), x)).strip(),
                    axis=1,
                )
                .tolist()
            )

            # # * Hexadecimal encoding
            # # iterate over the rows of the dataframe i.e. actions, each one is a triple of words
            # val = []
            # for action in df[["verb", "this", "that"]].values:
            #     count = 0
            #     # iterate over the words of the action
            #     for word in action:
            #         # 1. each word is a string, iterate over the characters
            #         # 2. get the sum of their ascii values
            #         # 3. convert them to hex
            #         # 4. append them to a list
            #         count += sum([int(ord(c)) for c in list(word)])
            #     val.append(hex(count).split("x")[-1])
            # self.data[f.stem] = val

    # def get_vocab(self) -> list:
    #     for f in self.files:
    #         df = pd.read_csv(f)
    #         words = df[["verb", "this", "that"]].values.flatten()
    #         for word in words:
    #             word = word.replace(" ", "")
    #             if word not in self.idx2word:
    #                 self.idx2word.append(word)

    #     for i, word in enumerate(self.idx2word):
    #         self.word2idx[word] = i

    # def get_vocab_action(self):
    #     vocab = []
    #     for f in self.files:
    #         df = pd.read_csv(f)
    #         actions = df[["verb", "this", "that"]]
    #         actions = actions.apply(lambda x: " ".join(x).strip(), axis=1).tolist()
    #         vocab.extend(actions)
    #     return list(set(vocab))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> list:
        key = self.files[index].stem
        val = self.data[key]
        return val

    def collate_fn(self, batch) -> dict:
        min_n = min([len(x) for x in batch]) - 1
        # get a random number between 0 and min_len - 1
        n = np.random.randint(1, min_n)
        hist, gt = [], []
        for x in batch:
            hist.append(x[:n])
            gt.append(x[n])

        out = {"hist": hist, "gt": gt}
        return out

    # ! Done for Assembly101 and not our version
    # def __getitem__(self, index) -> dict:
    #     id = self.files[index].split("_", 1)[-1].split(".")[0]
    #     return self.data[id]


if __name__ == "__main__":
    # * Dataset
    dataset = AssemblyTextDataset("data/mistake_labels", split="test")
    vocab = dataset.get_vocab()
    vocab = dataset.get_vocab_action()

    # * Model
    model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # * Dataloader
    # TODO don't know why but batch size higher than 2 gives problem to tokenization
    dl = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    for batch in dl:
        B = len(batch)
        hist = batch["hist"]
        gt = batch["gt"]

        # * BERT
        texts = []
        for entry in hist:
            text = " ".join(entry)
            text = text + 3 * " [MASK]"
            texts.append(text)

        # TODO maybe we should make it autoregressive to avoid predicting always the same word
        inputs = tokenizer(texts, return_tensors="pt")
        token_logits = model(**inputs).logits
        # Find the location of [MASK] and extract its logits
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)

        mask_token_logits = token_logits[mask_token_index[0], mask_token_index[1], :]
        N, D = mask_token_logits.shape
        mask_token_logits = mask_token_logits.reshape(B, N // B, D)
        # Pick the [MASK] candidates with the highest logits
        top_k_tokens = torch.topk(mask_token_logits, 1, dim=-1).indices.transpose(1, 2)

        for i, (text, k_tokens) in enumerate(zip(texts, top_k_tokens)):
            print(f"{i}: text")
            for tokens in k_tokens:
                print(
                    f'{i}: {text.replace("[MASK] [MASK] [MASK]", tokenizer.decode(tokens))}'
                )

        # TODO get only the predicted tokens and compare them to the gt
        # make_autoregressive(texts, n=3, model=model, tokenizer=tokenizer)

        # * GPT2
        # texts = [" ".join(entry) for entry in hist]

        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # model = GPT2LMHeadModel.from_pretrained("gpt2")

        # prompt = texts[0]
        # print(prompt)
        # input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        # pad_token_id = tokenizer.eos_token_id

        # sample_output = model.generate(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     pad_token_id=pad_token_id,
        #     do_sample=True,
        #     max_length=len(prompt.split()) + 3,
        #     top_k=len(prompt.split()),
        #     top_p=0.95,
        #     num_return_sequences=10,
        # )

        # for i, sample in enumerate(sample_output):
        #     generated_text = tokenizer.decode(sample, skip_special_tokens=True)
        #     generated_words = generated_text.split()[-3:]
        #     print(f"Variant {i+1}: {' '.join(generated_words)}")

        break
