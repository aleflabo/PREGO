import json
import os
import pickle
import re
from typing import Optional

import fire
import numpy as np
from ollama import ChatResponse, chat

BASE_PATH = "step_anticipation/data"
CONTEXT_PROMPT_PATH = f"{BASE_PATH}/context_prompt"
PREDICTIONS_PATH = f"{BASE_PATH}/predictions"


def get_metrics(preds, gts):
    tp, fp, fn, tn = 0, 0, 0, 0
    count, samples = 0, 0
    for k in gts.keys():
        gt = gts[k]
        pred = preds[k]
        matches = np.array([g in p for g, p in zip(gt, pred)])

        count += np.sum(matches)
        samples += len(matches)
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
    ollama_model,
    seq: list,
    prompt: str,
    toy: Optional[str],
    toy_class: Optional[str],
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

    for i in range(len(seq)):
        prompt_builder = load_data(f"{CONTEXT_PROMPT_PATH}/context_prompt.json")
        init = prompt_builder[prompt_context]["init"]

        if remove_toySequence:
            prompt_ = f"{prompt}{init} {toy_class}\n"
        else:
            prompt_ = f"{prompt}{init} {toy}\n"

        if type_prompt == "emoji":
            hist, action = ["👉"] + seq[:i], seq[i]
        else:
            hist, action = [-1] + seq[:i], seq[i]

        input_builder = prompt_builder[prompt_context]["input"]
        prompt_ += f"{input_builder}\n {', '.join(map(str, hist))}\n"

        output_builder = prompt_builder[prompt_context]["output"]
        prompt_ += f"{output_builder}\n"

        pred = set()

        messages = [
            {
                "role": "system",
                "content": "Always provide only the final output, consisting in one and only one number. Never output anything different from a single number.",
            },
            {
                "role": "user",
                "content": prompt_,
            },
        ]
        for _ in range(num_samples):
            response: ChatResponse = chat(model=ollama_model, messages=messages)
            # result = ollama_model.generate(
            #     prompt=prompt_,
            #     max_tokens=max_gen_len,
            #     temperature=temperature,
            #     top_p=top_p,
            # )

            result = response.message.content
            print(f"> {result}")
            v = result.strip().strip("_")
            # v = result.text.strip().strip("_")

            if type_prompt == "num":
                v = re.sub(r"^[^0-9]*|[^0-9]*$", "", v)
                try:
                    v = int(v)
                except:
                    pass

            pred.add(v if type_prompt != "emoji" else v[0] if v else "")

        gts.append(action)
        preds.append(pred)

    return preds, gts


# def anticipation(
#     seq: list,
#     prompt: str,
#     toy: Optional[str],
#     toy_class: Optional[str],
#     ollama_model,
#     max_gen_len: Optional[int],
#     temperature: float,
#     top_p: float,
#     num_samples: int,
#     clean_prediction: bool,
#     type_prompt="num",
#     prompt_context="default",
# ):
#     preds, gts = [], []

#     if type_prompt == "emoji":
#         prompt = prompt.replace("-1", "👉")

#     if toy_class:
#         remove_toySequence = True
#         prompt = remove_sequenceInput(prompt, toy_class)
#     else:
#         remove_toySequence = False

#     for i in range(len(seq)):
#         prompt_builder = load_data(f"{CONTEXT_PROMPT_PATH}/context_prompt.json")
#         init = prompt_builder[prompt_context]["init"]

#         if remove_toySequence:
#             prompt_ = f"{prompt}{init} {toy_class}\n"
#         else:
#             prompt_ = f"{prompt}{init} {toy}\n"

#         if type_prompt == "emoji":
#             hist, action = ["👉"] + seq[:i], seq[i]
#         else:
#             hist, action = [-1] + seq[:i], seq[i]

#         input_builder = prompt_builder[prompt_context]["input"]
#         prompt_ += f"{input_builder}\n {', '.join(map(str, hist))}\n"

#         output_builder = prompt_builder[prompt_context]["output"]
#         prompt_ += f"{output_builder}\n"

#         pred = set()
#         for _ in range(num_samples):
#             result = ollama_model.generate(
#                 prompt=prompt_,
#                 max_tokens=max_gen_len,
#                 temperature=temperature,
#                 top_p=top_p,
#             )

#             v = result.text.strip().strip("_")

#             if type_prompt == "num":
#                 v = re.sub(r"^[^0-9]*|[^0-9]*$", "", v)
#                 try:
#                     v = int(v)
#                 except:
#                     pass

#             pred.add(v if type_prompt != "emoji" else v[0] if v else "")

#         gts.append(action)
#         preds.append(pred)

#     return preds, gts


def main(
    ollama_model_name: str,
    # max_seq_len: int = 512,
    # max_gen_len: Optional[int] = None,
    # temperature: float = 0.6,
    # top_p: float = 0.9,
    ollama_model="llama3.2:3b",
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

    elif dataset == "epictent":
        contexts = load_data(
            f"{CONTEXT_PROMPT_PATH}/epictent_context_prompt_train.json"
        )

        if recognition_model == "OadTR":
            seqs = load_data(f"{PREDICTIONS_PATH}/output_OadTR_Epic-Tent-O.json")
        elif recognition_model == "miniROAD":
            seqs = load_data(f"{PREDICTIONS_PATH}/output_miniROAD_Epic-Tent-O_edo.json")
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    preds, gts = {}, {}

    for i, (k, v) in enumerate(seqs.items()):
        if dataset == "assembly":
            toy = get_toy(k)

            if toy_class_context:
                toy_class = toy2class[toy]
                prompt = contexts[toy_class][type_prompt]
            else:
                toy_class = None
                prompt = contexts.get(toy, {}).get(type_prompt, "")

        elif dataset == "epictent":
            toy = None
            toy_class = None
            prompt = contexts[type_prompt]
        else:
            raise ValueError(f"Dataset {dataset} not supported")

        seq = v["gt"] if use_gt else v["pred"]

        pred, gt = anticipation(
            seq=seq,
            prompt=prompt,
            toy=toy,
            toy_class=toy_class,
            ollama_model=ollama_model,
            # max_gen_len=max_gen_len,
            # temperature=temperature,
            # top_p=top_p,
            num_samples=num_samples,
            clean_prediction=clean_prediction,
            type_prompt=type_prompt,
            prompt_context=prompt_context,
        )

        preds[k] = pred
        gts[k] = gt

    if eval_metrics:
        metrics = get_metrics(preds=preds, gts=gts)
        print(f"[INFO] {metrics}")

    model_name = ollama_model_name.replace("/", "_")
    save_folder = f"results/{model_name}_{dataset}_{prompt_context}"

    os.makedirs(save_folder, exist_ok=True)
    pickle.dump(gts, open(f"{save_folder}/gts.pkl", "wb"))
    pickle.dump(preds, open(f"{save_folder}/preds.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire(main)
