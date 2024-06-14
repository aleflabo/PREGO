# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import datetime
import json
import math
import os
import random
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from ipdb import set_trace

import util
import utils

# from util.ClassificationMetrics import ClassificationMetricsEpoch


all_class_name = [
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
]


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    args=None,
):
    model.train()
    criterion.train()
    # metrics_calculator = ClassificationMetricsEpoch(num_classes=args.numclass)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 500
    num_class = args.numclass

    for (
        session,
        start,
        end,
        camera_inputs,
        motion_inputs,
        enc_target,
        distance_target,
        class_h_target,
        dec_target,
        # additional_class,
    ) in metric_logger.log_every(data_loader, print_freq, header):
        camera_inputs = camera_inputs.to(device)
        motion_inputs = motion_inputs.to(device)

        enc_target = enc_target.to(device)
        distance_target = distance_target.to(device)
        class_h_target = class_h_target.to(device)
        dec_target = dec_target.to(device)

        # !Uncomment
        if args.add_labels:
            additional_class = additional_class.to(device)

            enc_score_p0, dec_scores = model(
                camera_inputs, motion_inputs, additional_class
            )

        else:
            enc_score_p0, dec_scores = model(camera_inputs, motion_inputs)
        # enc_score_p0, dec_scores = \
        # model(camera_inputs, motion_inputs)

        # outputs = {
        #     'labels_encoder': enc_score_p0,  # [128, 298]
        #     'labels_decoder': dec_scores.view(-1, num_class),  # [128, 8, 298]

        # }
        # targets = {
        #     'labels_encoder': class_h_target.view(-1, num_class),
        #     'labels_decoder': dec_target.view(-1, num_class),

        # }

        outputs = {
            "labels_encoder": enc_score_p0,  # [128, 297]
        }
        targets = {
            "labels_encoder": class_h_target.view(-1, num_class),
        }
        #! calculate our statistics
        # mean_average_precision = metrics_calculator.update_batch(logits=outputs['labels_encoder'], targets=targets['labels_encoder'])

        loss_dict = criterion(outputs, targets)
        # loss_dict_decoder = criterion(outputs_decoder, targets_decoder)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # wandb.log({"loss": loss_value, "Epoch": epoch, "MAP": mean_average_precision})
        wandb.log({"loss": loss_value, "Epoch": epoch})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # !Added by us
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        # # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # MAP_epoch = metrics_calculator.statistics_epoch()
    # print('MAP_epoch: ', MAP_epoch)
    # wandb.log({"MAP_epoch": MAP_epoch})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
