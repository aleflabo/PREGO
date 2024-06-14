#!/usr/bin/env bash

PYTHONPATH=. python src/run.py --cfg/lstm.yaml --wandb-group 'forecasting' --wandb-name 'lstm' --wandb-tags 'train' --wandb-mode 'disabled'