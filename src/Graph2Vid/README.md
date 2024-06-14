Graph2Vid: Flow graph to Video Grounding for Weakly-supervised Multi-Step Localization
========

This is the official PyTorch implementation of Graph2Vid [1] (ECCV'22 oral).
The repo includes, the code to run graph grounding on the CrossTask dataset [2], as well as the corresponding flow graphs.

### Set up the data and the pre-trained models
1. Unpack the CrossTask data and the corresponding flow graphs by running the following command in the project root:
    ```unzip crosstask_with_graphs.zip```
    The folder contains flow graphs created i) manually, ii) obtained with the learning-based parser, or iii) the rule-based parser.
2. Git-clone the MIL-NCE [3] feature extractor from the [official repo](https://github.com/antoine77340/MIL-NCE_HowTo100M):
    ```git clone https://github.com/antoine77340/MIL-NCE_HowTo100M.git```
3. Set up the paths to where you git-cloned the MIL-NCE repo. For that, modify the `S3D_PATH` variable in `paths.py`.

### Run graph grounding on CrossTask
 1. Open the `evaluate.ipynb` notebook and run the step localization evaluation on CrossTask.


### Reference
[1] Dvornik et al. "Graph2Vid: Flow graph to Video Grounding for Weakly-supervised Multi-Step Localization." ECCV'22.

[2] Zhukov et al. "Cross-task: weakly supervised learning from instructional videos." CVPR'19

[3] Miech et al. "End-to-end learning of visual representations from uncurated instructional videos." CVPR'20.
