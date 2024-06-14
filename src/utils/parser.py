import argparse

# TODO: Add your name here
parser = argparse.ArgumentParser(description="PUT YOUR NAME HERE")

parser.add_argument(
    "--cfg",
    type=str,
    default="configs/default.yaml",
    help="YAML configuration file",
)

parser.add_argument("--debug", action="store_true", help="Debug mode")

# * WandB
parser.add_argument("--wandb-mode", type=str, default="disabled", help="WandB mode")
parser.add_argument("--wandb-group", type=str, default=None, help="WandB group")
parser.add_argument(
    "--wandb-name", type=str, default=None, required=True, help="WandB name"
)
parser.add_argument("--wandb-tags", type=str, default=None, help="WandB tags")
parser.add_argument("--wandb-notes", type=str, default=None, help="WandB notes")

# * TaskGraph
parser.add_argument("--hold-print", action="store_true", help="Hold print")
parser.add_argument(
    "--clustering-th", type=float, default=1.0, help="Clustering distance threshold"
)
parser.add_argument(
    "--match-th", type=float, default=0.46, help="Matching distance threshold"
)
parser.add_argument(
    "--beam-search-th", type=float, default=0.30, help="Beam search distance threshold"
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["coin", "crosstask", "assembly-label"],
    default="coin",
    help="Dataset to use",
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="/media/hdd/data/assembly101/data/annotations/",
    help="Dataset path",
)


parser.add_argument(
    "--eval-mode", type=str, choices=["text"], default="text", help="Evaluation mode"
)
parser.add_argument(
    "--graph-type", type=str, choices=["overall"], default="overall", help="Graph type"
)
parser.add_argument("--use-clusters", action="store_true", help="Use clusters")
parser.add_argument(
    "--method",
    type=str,
    choices=["beam-search-with-cluster", "baseline-with-cluster"],
    default="beam-search-with-cluster",
    help="Method to use",
)
parser.add_argument("--prune-keysteps", action="store_true", help="Prune keysteps")
parser.add_argument("--keysteps-th", type=float, default=0.0, help="Keysteps threshold")

# * BERT
parser.add_argument("--lm", type=str, default="bert", help="Language model")
parser.add_argument(
    "--mask-mode",
    type=str,
    default="none",
    choices=["none", "end", "prob"],
    help="Tokenize mode",
)
parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
parser.add_argument("--tokenize-prob", type=float, default=0.15, help="Tokenize prob")
parser.add_argument("--epochs", type=int, default=100, help="Epochs")
parser.add_argument("--validate-every", type=int, default=10, help="Validate every")

# * Misc
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="The GPU or CPU to use, standard PyTorch rules apply",
)

args = parser.parse_args()
