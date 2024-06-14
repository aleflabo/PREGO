import json
import os
import random
from typing import List, Optional, Tuple

import fire

from llama import Dialog, Llama
from llama.generation import Llama

# * Global variables
ROOT_FOLDER = "/media/hdd/usr/edo/egoProcel_mistakes/data/mistake_jsons_split"
# correct
CORRECT_JSON_FOLDER = os.path.join(ROOT_FOLDER, "correct")
CORRECT_JSON_FILES = os.listdir(CORRECT_JSON_FOLDER)
# mistake
MISTAKE_JSON_FOLDER = os.path.join(ROOT_FOLDER, "mistake")
MISTAKE_JSON_FILES = os.listdir(MISTAKE_JSON_FOLDER)


def extract_toy_name(example_name: str) -> str:
    """
    Extracts the toy name from the example name.

    Args:
        example_name (str): The name of the example.

    Returns:
        str: The name of the toy.
    """
    return example_name.split("_")[3].split("-")[1]


def modify_context_and_input_str(json_dict) -> Tuple[str, str]:
    pos_examples = json_dict["pos_examples"]
    toy_name = json_dict["toy"]
    toy_examples_names = [extract_toy_name(example) for example in pos_examples]
    context_str = json_dict["context_str"]
    input_str = json_dict["input_str"]
    input_str = f"Sequence type: {toy_name}\n" + input_str
    single_examples = context_str.split("---\n")[:-1]
    assert len(pos_examples) == len(single_examples)
    context_str = []
    for example, toy_example_name in zip(single_examples, toy_examples_names):
        curr_example_str = f"Sequence type: {toy_example_name}\n" + example
        context_str.append(curr_example_str)
    context_str = "---\n".join(context_str) + "---\n"

    return context_str, input_str


def truncated_strings(json_dict: dict) -> Tuple[List[str], List[str]]:
    all_truncated_prompts = []
    all_gts = []
    input_str = json_dict["input_str"]
    context = json_dict["context_str"]
    output_str = json_dict["output_str"]
    input_prompt, sequence_, output_prompt, _ = json_dict["input_str"].split("\n")
    sequence = sequence_.split(",")
    for i in range(len(sequence)):
        curr_str = (
            context
            + input_prompt
            + "\n"
            + ",".join(sequence[:i])
            + "\n"
            + output_prompt
            + "\n"
        )
        curr_res = sequence[i]
        all_truncated_prompts.append(curr_str)
        all_gts.append(curr_res)
    all_truncated_prompts.append(context + input_str)
    all_gts.append(output_str)
    return all_truncated_prompts, all_gts


def process_inputs_chat(data: str) -> List[Dialog]:
    """
    Processes the input string for chat completion.

    Args:
        data (str): The input string.

    Returns:
        List[Dialog]: The processed input string.
    """
    # * only user
    # user = {"role": "user", "content": data}
    # * user and system
    # system = {"role": "system", "content": "Always answer with the prediction only."}
    # assistant = {
    #     "role": "assistant",
    #     "content": "\n".join(data.split("\n")[:-5]),
    # }
    # return [[user]]
    # * user and assistant
    user, assistant = [], []
    for x in data.split("---"):
        sep = "Next Symbol:"
        q, a = x.strip("\n").split(sep)
        user.append(q + sep)
        assistant.append(a)

    out = []
    for pair in zip(user, assistant):
        out.extend(
            [
                {"role": "user", "content": pair[0]},
                {"role": "assistant", "content": pair[1]},
            ]
        )

    return [out[:-1]]


def eval(
    exp: str,
    mode: str,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    num_samples: int = 1,
) -> float:
    """
    Evaluates the model on the given mode.

    Args:
        exp (str): The experiment to evaluate. Can be one of "IS" or "LS".
        mode (str): The mode to evaluate the model on. Can be one of "all", "correct", or "mistake".
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (Optional[int], optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
        num_samples (int, optional): The number of samples to generate. Defaults to 1.

    Returns:
        float: The ratio of correct predictions.

    Raises:
        ValueError: If the mode is invalid.
    """
    # * Select jsons
    folder, files = None, None
    if mode == "all":
        raise NotImplementedError
    elif mode == "correct":
        folder = CORRECT_JSON_FOLDER
        files = CORRECT_JSON_FILES
    elif mode == "mistake":
        folder = MISTAKE_JSON_FOLDER
        files = MISTAKE_JSON_FILES
    else:
        raise ValueError("Invalid mode")

    chat = True if "chat" in ckpt_dir else False

    # * Load LLM
    # generator = Llama.build(
    #     ckpt_dir=ckpt_dir,
    #     tokenizer_path=tokenizer_path,
    #     max_seq_len=max_seq_len,
    #     max_batch_size=max_batch_size,
    # )

    # * Eval
    tot = 0
    correct = 0

    for json_file in files:
        print(json_file)

        with open(os.path.join(folder, json_file), "r") as f:
            curr_dict = json.load(f)

        if exp == "IS":
            sequences, gts = truncated_strings(curr_dict)
            for input_for_LLM, gt in zip(sequences, gts):
                tot += 1
                # * Generate
                if chat:
                    dialogs = process_inputs_chat(input_for_LLM)
                    results = generator.chat_completion(
                        dialogs,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    prompts = [input_for_LLM] * num_samples
                    results = generator.text_completion(
                        prompts,  # type: ignore
                        max_gen_len=max_gen_len,  # 4
                        temperature=temperature,
                        top_p=top_p,
                    )
                gt = gt.strip(" \n")

                preds = set()
                for res in results:
                    if chat:
                        preds.add(res["generation"]["content"].strip(" \n"))
                    else:
                        preds.add(res["generation"].strip(" \n"))

                print(
                    "Procedure Label: {}\nGT: {}\tPred: {}\nCorrect: {}\n".format(
                        mode, gt, preds, gt in preds
                    )
                )

                if gt in preds:
                    correct += 1

        elif exp == "LS":
            tot += 1
            context_str, input_str = modify_context_and_input_str(curr_dict)
            input_for_LLM = context_str + input_str

            # * Generate
            if chat:
                dialogs = process_inputs_chat(input_for_LLM)
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                prompts = [input_for_LLM] * num_samples
                results = generator.text_completion(
                    prompts,  # type: ignore
                    max_gen_len=max_gen_len,  # 4
                    temperature=temperature,
                    top_p=top_p,
                )

            # Output
            gt = curr_dict["output_str"].strip(" \n")
            preds = set()
            for res in results:
                if chat:
                    preds.add(res["generation"]["content"].strip(" \n"))
                else:
                    preds.add(res["generation"].strip(" \n"))

            print(
                "Procedure Label: {}\nGT: {}\tPred: {}\nCorrect: {}\n".format(
                    curr_dict["procedure_label"], gt, preds, gt in preds
                )
            )
            if gt in preds:
                correct += 1
        else:
            raise ValueError("Invalid experiment")

    ratio = correct / tot
    print("Ratio:", ratio, f"{correct}/{tot}")
    return ratio


if __name__ == "__main__":
    # * Run LLAMA-2-7B-chat on mistake jsons
    # $ torchrun --nproc_per_node 1 run_llama.py --ckpt_dir /media/ssd/usr/edo/llama/llama-2-7b-chat/ --tokenizer_path /media/ssd/usr/edo/llama/tokenizer.model --max_seq_len 2048 --max_batch_size 6 --mode mistake --exp LS
    # * Run LLAMA-2-13B on correct jsons
    # $ torchrun --nproc_per_node 2 run_llama.py --ckpt_dir /media/ssd/usr/edo/llama/llama-2-13b/ --tokenizer_path /media/ssd/usr/edo/llama/tokenizer.model --max_seq_len 2048 --max_batch_size 6 --mode mistake --exp LS --num_samples 5 --max_gen_len 4
    # ! you can remove `eval` from within `fire.Fire(eval)` and call the function as an argument in the command line, I have yet
    # ! to find a way to do it in the debugger
    # $ torchrun --nproc_per_node 1 run_llama.py eval --ckpt_dir /media/ssd/usr/edo/llama/llama-2-7b/ --tokenizer_path /media/ssd/usr/edo/llama/tokenizer.model --max_seq_len 2048 --max_batch_size 6 --mode correct --exp LS --num_samples 5 --max_gen_len 4
    fire.Fire(eval)
