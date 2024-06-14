import json
import pdb
from ast import Tuple
from tempfile import tempdir
from typing import Optional

import fire
import numpy as np

from llama import Dialog, Llama
from llama.generation import Llama


def get_toy(name: str) -> str:
    """Get the toy name from the file name

    Args:
        name (str): file name

    Returns:
        str: toy name
    """
    toy = name.split("-")[2].split("_")[0]
    return toy


def load_data(path: str) -> dict:
    """
    Load the data from the json file

    Args:
        path (str): path to the json file

    Returns:
        dict: dictionary with the data
    """
    data = json.load(open(path, "r"))
    return data


def load_vocab(vocab_file: str) -> list:
    vocab = np.loadtxt(vocab_file, dtype=str)
    vocab = vocab[:, 1].tolist()
    return vocab


def anticipation(
    seq: list,
    toy: str,
    skip_n: int,
    llm,
    max_gen_len: Optional[int],
    temperature: float,
    top_p: float,
    num_samples: int = 1,
    vocab_file: Optional[str] = None,
):
    global prompt
    global preds, gts

    vocab = load_vocab(vocab_file) if vocab_file is not None else None

    # iterate over the sequence
    for i in range(len(seq)):
        # Skip the first element to avoid empy context
        if i < skip_n:
            continue
        prompt += f"Sequence type: {toy}\n"
        hist, action = seq[:i], seq[i]
        hist = [-1] if len(hist) == 0 else hist
        # Add the history
        prompt += f"Input Sequence:\n {', '.join(map(str,hist))}\n"
        # Add the action
        prompt += f"Next Symbol:\n"
        # LLM
        prompts = [prompt] * num_samples
        # predict the next symbol with LLAMA2
        results = llm.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        pred = set()
        for res in results:
            v = res["generation"].strip(" \n")

            try:
                v = int(v)
                v = vocab[v] if vocab is not None else v
            except:
                pass

            pred.add(v)

        prompt += f" {action}\n---\n"
        gts.append(action)
        preds.append(pred)
        print(f"[INFO]\taction: {action} in {pred} >>> {action in pred}")
    return preds, gts


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    temperature: float = 0.6,
    top_p: float = 0.9,
    num_samples: int = 1,
    skip_n: int = 1,
    context_splits: int = 11,
    vocab_file: Optional[str] = None,
    use_gt: bool = False,
):
    # Placeholder tensor of shape BxSxD
    seqs = load_data("results/edo.json")

    # * Global variables
    global prompt
    global preds, gts
    prompt = ""
    preds, gts = [], []

    llm = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # * Split size
    # Due to memory constraints we cannot have a global context,
    # so we split it in `context_splits` parts.
    max_split = (
        len(seqs) // context_splits
    )  # ! different sequences may have different lenghts

    count, samples, split_counter = 0, 0, 0
    for i, (k, v) in enumerate(seqs.items()):
        toy = get_toy(k)
        print(f"[INFO]\t{i}/{len(seqs)}>>>{toy}")

        seq = v["gt"] if use_gt else v["pred"]

        preds, gts = anticipation(
            seq=seq,
            toy=toy,
            skip_n=skip_n,
            llm=llm,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
            vocab_file=vocab_file,
        )

        # pdb.set_trace()

        matches = [int(g in p) for p, g in zip(preds, gts)]
        count += sum(matches)
        samples += len(matches)

        if split_counter == max_split:
            split_counter = 0
            prompt = ""

        split_counter += 1

    # pdb.set_trace()
    ratio = count / samples
    print(f"[INFO]\t{ratio}({count}/{samples})")
    return ratio


if __name__ == "__main__":
    fire.Fire(main)
