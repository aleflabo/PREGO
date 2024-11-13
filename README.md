# PREGO: online mistake detection in PRocedural EGOcentric videos (CVPR 2024)

```diff 
-**[Project Page](https://www.pinlab.org/prego)**
``` 
|
**[PREGO paper [CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/html/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.html)** 
|
**[TI-PREGO paper [arXiv]](https://arxiv.org/abs/2411.02570)**


## Index

1. [Introduction](#introduction)
2. [News](#news)
3. [Preparation](#preparation)
    - [Data](#data)
    - [LLAMA](#llama)
    - [Environment](#environment)
4. [Usage](#usage)
    - [Step Recognition](#step-recognition)
    - [Data Aggregation](#data-aggregation)
    - [Step Anticipation](#step-anticipation)
        - [Data Preparation](#data-preparation)
        - [Parameters](#parameters)
        - [Run](#run)
5. [Reference](#reference)


## Introduction
This repo hosts the official PyTorch implementations of the *IEEE/CVF Computer Vision and Pattern Recognition (CVPR) '24* paper **PREGO: online mistake detection in PRocedural EGOcentric videos** and of the follow-up paper **TI-PREGO: Chain of Thought and In-Context Learning for Online Mistake Detection in PRocedural EGOcentric Videos**.

PREGO is the first online one-class classification model for mistake detection in procedural egocentric videos. It uses an online action recognition component to model current actions and a symbolic reasoning module to predict next actions, detecting mistakes by comparing the recognized current action with the expected future one. We evaluate this on two adapted datasets, *Assembly101-O* and *Epic-tent-O*, for online benchmarking of procedural mistake detection.

![teaser_image](assets/teaser.png)

## News
 **[2024-11-12]** Uploaded the TSN features for Assembly101-O and Epic-tent-O [[GDrive]](https://drive.google.com/drive/u/1/folders/1gcOIEXhwysCE2o8-5C4vQnTShJ7p3CKH).

 **[2024-11-04]** Published the follow-up paper [[TI-PREGO]](https://arxiv.org/abs/2411.02570).
 
 **[2024-06-20]** Presented PREGO at #CVPR2024.
 
 **[2024-06-16]** Uploaded the anticipation branch.

## Preparation
### Data
```diff
- Work in progress
```

### LLAMA
In order to run our anticipation step with LLAMA, you have to be granted access to the models by Meta [here](https://www.llama.com/llama-downloads/).
Place them whenever you like, just recall to update the paths whenever necessary, as in `step_anticipation/scripts/anticipation.sh`.

### Environment
You can choose between creating a `conda` or `virtualenv` environment, as you prefer 
```bash 
# conda
conda create -n prego python
conda activate prego

# virtualenv
python -m venv .venv
source .venv/bin/activate
```
Then, install the requirements
```bash
pip install -r requirements.txt
```

```diff
- unsloth
```


## Usage

### Step Recognition
```diff
- Work in progress
```

### Data Aggregation
```diff
- Work in progress
```

### Step Anticipation

#### Data Preparation 
Description of the steps needed to prepare the data for the Step Anticipation branch. 

Step Recognition predictions: 
- place the predictions (after aggregation) of the Step Recognizer in the `step_anticipation/data/predictions` 
- the file should have the following structure: 
```json
{
    "nusar-2021_action_both_9044-a08_9044_user_id_2021-02-05_154403": {
        "pred": [
            39,
            37,
            74,
            39,
            37
        ],
        "gt": [
            37,
            80,
            39,
            29,
            85
        ]
    },
...
}
```
Context prompt:
- `step_anticipation/data/context_prompt/assembly_context_prompt_train.json` and `step_anticipation/data/context_prompt/epictents_context_prompt_train.json` contain the context to be used for the In-context learning prompt.
- `step_anticipation/data/context_prompt/context_prompt.json` contains the strings to fill the context prompt. 

#### Parameters
Description of the parameters that can be added to the `step_anticipation/scripts/anticipation.sh` script. 

- `ckpt_dir=/path/to//llama/llama-2-7b` 
- `tokenizer_path=/path/to/tokenizer/llama/tokenizer.model`
- `max_seq_len=2048` Maximum sequence length for input text
- `max_batch_size` Maximum batch size for generating sequences
- `temperature` Temperature value for controlling randomness in generation
- `max_gen_len` Maximum length of the generated text sequence.
- `num_samples` How many generations per each input context
- `use_gt` Select if gt or predictions from Step Recognizer are used as input context
- `dataset` Select dataset to use. ['assembly', 'epictent']
- `type_prompt` Select which type of context to be passed. ['num', 'alpha', 'emoji']
- `toy_class_context` For the assembly dataset only. If True, the input context has all the example from the same class of toys
- `recognition_model` If not use_gt, select which Step Recognizer predictions to use. ['miniROAD', 'OadTR']
- `prompt_context` Select how the prompt context is structured. ['default', 'unreferenced','elaborate','no-context']


#### Run
```bash
cd step_anticipation
./scripts/anticipation.sh
```

## Reference
If you find our code or paper to be helpful, please consider citing:
```
@InProceedings{Flaborea_2024_CVPR,
    author    = {Flaborea, Alessandro and di Melendugno, Guido Maria D'Amely and Plini, Leonardo and Scofano, Luca and De Matteis, Edoardo and Furnari, Antonino and Farinella, Giovanni Maria and Galasso, Fabio},
    title     = {PREGO: Online Mistake Detection in PRocedural EGOcentric Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18483-18492}
}
```
```
@misc{plini2024tipregochainthoughtincontext,
      title={TI-PREGO: Chain of Thought and In-Context Learning for Online Mistake Detection in PRocedural EGOcentric Videos}, 
      author={Leonardo Plini and Luca Scofano and Edoardo De Matteis and Guido Maria D'Amely di Melendugno and Alessandro Flaborea and Andrea Sanchietti and Giovanni Maria Farinella and Fabio Galasso and Antonino Furnari},
      year={2024},
      eprint={2411.02570},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.02570}, 
}
```
