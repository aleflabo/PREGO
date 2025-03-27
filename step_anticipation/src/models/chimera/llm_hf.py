import json
import os
import pickle
import re
import time
from typing import Optional

import transformers

transformers.logging.set_verbosity_error()
import fire
import ipdb
import numpy as np
import torch
from transformers import pipeline

BASE_PATH = "step_anticipation/data"
CONTEXT_PROMPT_PATH = f"{BASE_PATH}/context_prompt"
PREDICTIONS_PATH = f"{BASE_PATH}/predictions"

TIME_CNT = []


class HFModel:
    def __init__(
        self, model_name: str, max_seq_len: int = 512, max_batch_size: int = 8
    ):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            # max_batch_size=max_batch_size,
            # max_seq_len=max_seq_len,
            device_map="auto",
        )

    def text_completion(
        self, prompts, max_gen_len: Optional[int], temperature: float, top_p: float
    ):
        generate_kwargs = {
            "max_new_tokens": max_gen_len,
            # "max_gen_len": max_gen_len,
            "temperature": temperature,
            "top_p": top_p,
        }

        start_time = time.time()
        outputs = self.pipe(prompts, **generate_kwargs)
        TIME_CNT.append(time.time() - start_time)

        # Flatten the outputs (each prompt returns a list of outputs)
        flattened = []
        for res in outputs:
            if isinstance(res, list):
                flattened.append(res[0])
            else:
                flattened.append(res)
        return flattened


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
        correct = matches[:-1]
        mistake = matches[-1]

        tn += np.sum(correct)
        fp += np.sum(~correct)
        tp += int(not mistake)
        fn += int(mistake)

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
    toy = name.split("-")[2].split("_")[0]
    return toy


def load_data(path: str) -> dict:
    data = json.load(open(path, "r"))
    return data


def remove_sequenceInput(prompt, toy_class):
    new_prompt = ""
    start = 0
    for m in re.finditer(r"Sequence type: [a-zA-Z0-9]{3,}\n", prompt):
        new_prompt += prompt[start : m.start()]
        new_prompt += f"Sequence type: {toy_class}\n"
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
        prompt = prompt.replace("-1", "👉")

    if toy_class:
        remove_toySequence = True
        prompt = remove_sequenceInput(prompt, toy_class)
    else:
        remove_toySequence = False

    # ! Don't know why it was inside the loop
    # >>>
    prompt_builder = load_data(f"{CONTEXT_PROMPT_PATH}/context_prompt.json")
    init = prompt_builder[prompt_context]["init"]

    if remove_toySequence:
        prompt_ = f"{prompt}{init} {toy_class}\n"
    else:
        prompt_ = f"{prompt}{init} {toy}\n"
    # <<<

    for i in range(len(seq)):
        if type_prompt == "emoji":
            hist, action = ["👉"] + seq[:i], seq[i]
        else:
            hist, action = [-1] + seq[:i], seq[i]

        # initialize history if empty
        if type_prompt == "emoji":
            hist = ["👉"] if len(hist) == 0 else hist
        else:
            hist = [-1] if len(hist) == 0 else hist

        print(f"[INFO] >>> {hist} -> {action}")

        input_builder = prompt_builder[prompt_context]["input"]
        prompt_ += f"{input_builder}\n {', '.join(map(str, hist))}\n"

        output_builder = prompt_builder[prompt_context]["output"]
        prompt_ += f"{output_builder}\n"

        pred = set()
        for _ in range(num_samples):
            prompts = [prompt_] * num_samples

            # NEW: Using our HFModel text_completion method (which calls the Hugging Face pipeline)
            results = llm.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for res in results:
                res = res["generated_text"].replace(prompt_, "")  # Remove the prompt
                # v = re.sub(r"^[ \n\.,;:]+|[ \n\.,;:]+$", "", res)
                v = re.sub(r"[ \n\.,;:]+", "", res)
                v = v.strip("_")

                if type_prompt == "num":
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
        gts.append(action)
        preds.append(pred)
        print(f"[INFO] >>>> {action} in {pred} ---> {action in pred}")

    return preds, gts


def main(
    model_name: str,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = 20,
    temperature: float = 0.6,
    top_p: float = 0.9,
    num_samples: int = 1,
    use_gt: bool = False,
    type_prompt: str = "num",
    clean_prediction: bool = False,
    eval_metrics: bool = True,
    dataset: str = "assembly",
    toy_class_context: bool = False,
    recognition_model: str = "miniROAD",
    prompt_context: str = "default",
):
    if dataset == "assembly":
        if toy_class_context:
            toy2class = json.load(open("assets/toy2class.json", "r"))
            contexts = load_data(
                f"{CONTEXT_PROMPT_PATH}/assembly_context_prompt_train.json"
            )
        else:
            contexts = load_data(
                f"{CONTEXT_PROMPT_PATH}/supplementary/assembly_context_prompt_train_onlyToy.json"
            )

        if recognition_model == "OadTR":
            seqs = load_data(f"{PREDICTIONS_PATH}/output_OadTR_Assembly101-O.json")
        elif recognition_model == "miniROAD":
            seqs = load_data(f"{PREDICTIONS_PATH}/output_miniROAD_Assembly101-O.json")

        if type_prompt == "alpha":
            # load the idx2action mapping
            idx2action = pickle.load(open(f"{BASE_PATH}/idx2action.pkl", "rb"))
        elif type_prompt == "emoji":
            # load the idx2action mapping
            idx2emoji = json.load(open(f"{BASE_PATH}/idx2emoji.json", "r"))

    elif dataset == "epictent":
        contexts = load_data(
            f"{CONTEXT_PROMPT_PATH}/epictent_context_prompt_train.json"
        )

        if recognition_model == "OadTR":
            seqs = load_data(f"{PREDICTIONS_PATH}/output_OadTR_Epic-Tent-O.json")
        elif recognition_model == "miniROAD":
            seqs = load_data(f"{PREDICTIONS_PATH}/output_miniROAD_Epic-tent-O.json")
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    preds, gts = {}, {}
    global out_plot
    out_plot = {}

    llm = HFModel(
        model_name=model_name, max_seq_len=max_seq_len, max_batch_size=max_batch_size
    )

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

    model_identifier = model_name.split("/")[-1]  # Use model name for folder naming
    save_folder = "{}_{:d}_{}_{:d}_{:d}_{:.2f}_{}_{}".format(
        model_identifier,
        use_gt,
        type_prompt,
        int(clean_prediction),
        num_samples,
        temperature,
        dataset,
        prompt_context,
    )

    if not os.path.exists(f"results/{save_folder}"):
        os.makedirs(f"results/{save_folder}")

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
        print(f"Average time: {sum(TIME_CNT) / len(TIME_CNT)}")

    pickle.dump(gts, open(f"results/{save_folder}/hf_gts.pkl", "wb"))
    pickle.dump(preds, open(f"results/{save_folder}/hf_preds.pkl", "wb"))
    pickle.dump(out_plot, open(f"results/{save_folder}/plot.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire(main)
