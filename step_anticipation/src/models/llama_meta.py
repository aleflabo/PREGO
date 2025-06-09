import json
import os
import pickle
import re
from tempfile import tempdir
from typing import Optional

import fire
import numpy as np
from llama import Dialog, Llama
from llama.generation import Llama


def get_metrics(preds, gts):
    tp, fp, fn, tn = 0, 0, 0, 0
    count, samples = 0, 0
    for k in gts.keys():
        gt = gts[k]
        pred = preds[k]
        matches = np.array([g in p for g, p in zip(gt, pred)])

        count += np.sum(matches)
        samples += len(matches)
        # the last one is a mistake, a mismatch is expected
        # all the actions all correct procedures except the last one
        correct = matches[:-1]
        mistake = matches[-1]

        # tn: correct seen as correct
        tn += np.sum(correct)
        # fp: correct seen as mistake
        fp += np.sum(~correct)
        # tp: mistake seen as mistake
        tp += int(not mistake)
        # fn: mistake seen as correct
        fn += int(mistake)

    # * metrics
    # accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    ratio = count / samples

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ratio": ratio,
        "count": count,
        "samples": samples,
    }


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


def remove_sequenceInput(prompt, toy_class):
    # if toy_class is not None, remove the reference to the single toy class, i.e. "a21" and replace with the superclass, i.e. "dumper"
    new_prompt = ""
    start = 0
    count = 1
    for m in re.finditer(r"Sequence type: [a-zA-Z0-9]{3,}\n", prompt):
        new_prompt += prompt[start : m.start()]
        new_prompt += f"Sequence type: {toy_class}\n"
        count += 1
        start = m.end()
    new_prompt += prompt[start:]
    return new_prompt.replace("Symbol", "Sequence")


def anticipation(
    seq: list,
    prompt: str,
    toy: Optional[str],
    toy_class: Optional[str],
    llm,
    max_gen_len: Optional[int],
    temperature: float,
    top_p: float,
    num_samples: int,
    clean_prediction: bool,
    type_prompt="num",
    prompt_context="default",
):
    preds, gts = [], []

    if type_prompt == "emoji":
        # replacing the start of the sequence "-1" with an emoji
        prompt = prompt.replace("-1", "👉")

    if toy_class:
        remove_toySequence = True
        prompt = remove_sequenceInput(prompt, toy_class)
    else:
        remove_toySequence = False

    # iterate over the sequence
    for i in range(len(seq)):
        prompt_builder = load_data("data/context_prompt/context_prompt.json")
        init = prompt_builder[prompt_context]["init"]

        if remove_toySequence:
            prompt_ = f"{prompt}{init} {toy_class}\n"
        else:
            prompt_ = f"{prompt}{init} {toy}\n"

        if type_prompt == "emoji":
            hist, action = ["👉"] + seq[:i], seq[i]
        else:
            hist, action = [-1] + seq[:i], seq[i]

        # hist, action = seq[-2:-1], seq[-1] # !LS

        # initialize the history with a starting num or emoji
        if type_prompt == "emoji":
            hist = ["👉"] if len(hist) == 0 else hist
        else:
            hist = [-1] if len(hist) == 0 else hist

        print(f"[INFO] >>> {hist} -> {action}")

        # Add the history
        input_builder = prompt_builder[prompt_context]["input"]
        prompt_ += f"{input_builder}\n {', '.join(map(str,hist))}\n"

        # Add the action
        output_builder = prompt_builder[prompt_context]["output"]
        prompt_ += f"{output_builder}\n"

        # LLM
        pred = set()
        for sample in range(num_samples):

            # if needed multiple predictions for the same prompt
            prompts = [prompt_] * num_samples

            # predict the next symbol with LLAMA2
            results = llm.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            # pdb.set_trace()
            for res in results:
                # replace the following patterns at the beginning and end of the string:
                # - whitespaces
                # - newlines
                # - puntuaction
                v = re.sub(r"^[ \n\.,;:]+|[ \n\.,;:]+$", "", res["generation"])
                v = res["generation"].strip("_")

                if type_prompt == "num":
                    # remove non numeric characters from left and right
                    v = re.sub(r"^[^0-9]*|[^0-9]*$", "", v)
                    try:
                        v = int(v)
                    except:
                        pass

                if len(hist) in out_plot:
                    out_plot[len(hist)]["sum"] += len(pred)
                    out_plot[len(hist)]["count"] += 1
                else:
                    out_plot[len(hist)] = {"sum": len(pred), "count": 1}

                if type_prompt == "num":
                    pred.add(v)
                elif type_prompt == "emoji":
                    try:
                        pred.add(v[0])
                    except:
                        pred.add("")
                else:
                    pred.add(v[: v.find("\n")])
                # for p in re.findall(r'\d+', v):
                #     pred.add(int(p))

                # for p in re.findall(r"[^\w\s,]", v):
                #     pred.add(p)

        gts.append(action)
        preds.append(pred)
        print(f"[INFO] >>>> {action} in {pred} ---> {action in pred}")

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
    use_gt: bool = False,
    type_prompt: str = "num",
    clean_prediction: bool = False,
    eval_metrics: bool = True,
    dataset: str = "assembly",
    toy_class_context: bool = False,
    recognition_model: str = "miniROAD",  # select which recognition model to use. ["OadTR", "miniROAD"]
    prompt_context: str = "default",  # select which context prompt to use. ["default", "unreferenced", "elaborate", "no-context"]
):

    if dataset == "assembly":

        if toy_class_context:
            # load the same toy_class, i.e. "a01": "excavator", as context
            toy2class = json.load(open("assets/toy2class.json", "r"))
            contexts = load_data(
                "data/context_prompt/assembly_context_prompt_train.json"
            )
        else:
            # load only the same toy examples as context
            contexts = load_data(
                "data/context_prompt/supplementary/assembly_context_prompt_train_onlyToy.json"
            )

        if recognition_model == "OadTR":
            # using the predictions from the OadTR recognition model
            seqs = load_data("data/predictions/output_OadTR_Assembly101-O.json")
        elif recognition_model == "miniROAD":
            # using the predictions from the miniROAD recognition model
            seqs = load_data("data/predictions/output_miniROAD_Assembly101-O.json")

        if type_prompt == "alpha":
            # load the idx2action mapping
            idx2action = pickle.load(open("data/idx2action.pkl", "rb"))
        elif type_prompt == "emoji":
            # load the idx2action mapping
            idx2emoji = json.load(open("data/idx2emoji.json", "r"))

    elif dataset == "epictent":

        contexts = load_data("data/context_prompt/epictent_context_prompt_train.json")

        if recognition_model == "OadTR":
            # using the predictions from the OadTR recognition model
            seqs = load_data("data/predictions/output_OadTR_Epic-Tent-O.json")
        elif recognition_model == "miniROAD":
            # using the predictions from the miniROAD recognition model
            seqs = load_data("data/output_miniROAD_Epic-Tent-O_edo.json")

        if type_prompt == "emoji":
            idx2emoji = json.load(open("data/idx2emoji.json", "r"))

    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # * Global variables
    preds, gts = {}, {}
    global out_plot
    out_plot = {}

    llm = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # * Split size
    # Due to memory constraints we cannot have a global context,
    # so we split it in `context_splits` parts.
    for i, (k, v) in enumerate(seqs.items()):
        if dataset == "assembly":
            toy = get_toy(k)
            print(f"[INFO] > {i}/{len(seqs)}: {toy}")

            if toy_class_context:
                toy_class = toy2class[toy]
                prompt = contexts[toy_class][type_prompt]
            else:
                toy_class = None
                try:
                    prompt = contexts[toy][type_prompt]
                except:
                    prompt = ""

        elif dataset == "epictent":
            toy = None
            toy_class = None
            prompt = contexts[type_prompt]
            print(f"[INFO] > {i}/{len(seqs)}")
        else:
            raise ValueError(f"Dataset {dataset} not supported")

        seq = v["gt"] if use_gt else v["pred"]

        print(f"[INFO] >> {seq}")

        # convert action numbers to string or emoji if requested
        if type_prompt == "alpha" and dataset == "assembly":
            seq = [idx2action[s] for s in seq]
        elif type_prompt == "emoji":
            seq = [idx2emoji[str(s)]["escape"] for s in seq]

        pred, gt = anticipation(
            seq=seq,
            prompt=prompt,
            toy=toy,
            toy_class=toy_class,
            llm=llm,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
            clean_prediction=clean_prediction,
            type_prompt=type_prompt,
            prompt_context=prompt_context,
        )

        preds[k] = pred
        gts[k] = gt

        matches = [int(g in p) for p, g in zip(pred, gt)]

    # save preds and gts in pickle
    model = os.path.basename(ckpt_dir).split("-")[-1]
    save_folder = "{}_{:d}_{}_{:d}_{:d}_{:.2f}_{}_{}".format(
        model,
        use_gt,
        type_prompt,
        clean_prediction,
        num_samples,
        temperature,
        dataset,
        prompt_context,
    )

    if not os.path.exists(f"results/{save_folder}"):
        os.makedirs(f"results/{save_folder}", exist_ok=True)

    if eval_metrics:
        metrics = get_metrics(preds=preds, gts=gts)
        print(f"[INFO] {metrics}")
        print(
            "Ratio: {:.3f}\t({:d}/{:d})".format(
                metrics["ratio"], metrics["count"], metrics["samples"]
            )
        )
        print(
            "TP: {:d}, FP: {:d}, FN: {:d}, TN: {:d}".format(
                metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
            )
        )
        print(
            "Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )
        )
    pickle.dump(gts, open(f"results/{save_folder}/llama_gts.pkl", "wb"))
    pickle.dump(preds, open(f"results/{save_folder}/llama_preds.pkl", "wb"))
    pickle.dump(out_plot, open(f"results/{save_folder}/plot.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire(main)
