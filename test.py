import json
import math
import os
import pickle
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from ipdb import set_trace
from tqdm import tqdm

import util
import utils
from util.ClassificationMetrics import ClassificationMetricsEpoch

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


@torch.no_grad()
def test_one_epoch(
    model, criterion, data_loader, device, logger, args, epoch, nprocs=4
):
    model.eval()
    criterion.eval()
    metrics_calculator = ClassificationMetricsEpoch(
        num_classes=args.numclass, device=device
    )
    with open("./data/data_info_new.json", "r") as outfile:
        data_json = json.load(outfile)
    all_class_name = data_json["assembly"]["class_index"][1:]

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # header = 'Test:'
    # all_probs, all_classes = [], []
    # dec_score_metrics = []
    # dec_score_metrics_every = {}
    # dec_target_metrics_every = {}
    # num_query = args.query_num
    # !For Decoder
    # for iii in range(num_query):
    #     dec_score_metrics_every[str(iii)] = []
    #     dec_target_metrics_every[str(iii)] = []

    # dec_target_metrics = []

    num_class = args.numclass
    feat_type = args.feature
    video_names = []
    start_frame = []
    end_frame = []
    session_old = None
    encoding_scores = []
    session_list = []
    all_probs, all_classes = [], []

    # use additional class token
    use_additional_token = False

    # add additional class token to for cycle if you want train instead of test
    for (
        session,
        start,
        end,
        camera_inputs_val,
        motion_inputs_val,
        enc_target_val,
        distance_target_val,
        class_h_target_val,
        dec_target,
    ) in tqdm(
        data_loader
    ):  #! metric_logger.log_every(data_loader, 500, header):
        if use_additional_token:
            session_new = session[0]
            if session_new != session_old:
                session_old = session_new
                encoder_score_p0 = None
                additional_token = torch.zeros((1, num_class)).to(device)
                additional_token_tensor = None
            # if True: #use_additional_token:

            # control on additional class token
            if start > 64:
                # select only token that can be used
                additional_token_portion = additional_token_tensor[
                    -1
                ]  # [:start]   #decide whether to use all the past or just the last
                additional_token = torch.mean(additional_token_portion, dim=0)
                additional_token = torch.reshape(
                    additional_token_portion, (1, num_class)  # additional_token,
                )

        camera_inputs = camera_inputs_val.to(device)
        motion_inputs = motion_inputs_val.to(device)

        video_names += session
        start_frame += start
        end_frame += end

        class_h_target = class_h_target_val.to(device)
        """
        enc_target = enc_target_val.to(device)
        distance_target = distance_target_val.to(device)
        class_h_target = class_h_target_val.to(device)
        dec_target = dec_target.to(device)
        """
        enc_score_p0, dec_scores = model(
            camera_inputs, motion_inputs
        )  # ,additional_token)

        if use_additional_token:
            index = torch.argmax(enc_score_p0, dim=1)
            one_hot_class = torch.zeros((1, num_class)).to(device)
            one_hot_class[0, index] = 1
            additional_token_tensor = (
                torch.cat((additional_token_tensor, one_hot_class), dim=0)
                if additional_token_tensor is not None
                else one_hot_class
            )
        # outputs = {
        #     'labels_encoder': enc_score_p0,  # [128, 22]
        #     'labels_decoder': dec_scores.view(-1, num_class),  # [128, 8, 22]

        # }
        # targets = {
        #     'labels_encoder': class_h_target.view(-1, num_class),
        #     'labels_decoder': dec_target.view(-1, num_class),

        # }

        outputs = {
            "labels_encoder": enc_score_p0,  # [128, 144] #298]
        }
        targets = {
            "labels_encoder": class_h_target.view(-1, num_class),
        }
        metrics_calculator.update_batch(
            logits=outputs["labels_encoder"], targets=targets["labels_encoder"]
        )
        # print('MAP_precision_batch: ', MAP_precision_batch, 'average_precision_batch: ', average_precision_batch)
        #! calculate our statistics
        # accuracy, precision, recall, f1_score, auc_roc, MAP =
        metrics_calculator.update_batch(
            logits=outputs["labels_encoder"], targets=targets["labels_encoder"]
        )

        #
        loss_dict = criterion(outputs, targets)
        # loss_dict_decoder = criterion(outputs_decoder, targets_decoder)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }

        # metric_logger.update(
        #     loss=sum(loss_dict_reduced_scaled.values()),
        #     **loss_dict_reduced_scaled,
        #     **loss_dict_reduced_unscaled,
        # )

        # prob_val = enc_score_p0.cpu().numpy()

        prob_val = F.softmax(enc_score_p0, dim=-1).cpu().numpy()
        # prob_val = F.softmax(enc_score_p0, dim=-1).cpu().numpy()
        all_probs += list(prob_val)  # (89, 21)
        # dec_score_metrics = dec_scores.cpu().numpy()
        # dec_score_metrics += list(
        #     F.softmax(dec_scores.view(-1, num_class), dim=-1).cpu().numpy()
        # )
        # dec_score_metrics += list(F.softmax(dec_scores.view(-1, num_class), dim=-1).cpu().numpy())
        t0_class_batch = class_h_target.cpu().numpy()
        all_classes += list(t0_class_batch)
        # dec_target_metrics += list(dec_target.view(-1, num_class).cpu().numpy())
        # for iii in range(num_query):
        #     dec_score_metrics_every[str(iii)] += list(
        #         F.softmax(dec_scores[:, iii, :].view(-1, num_class), dim=-1)
        #         .cpu()
        #         .numpy()
        #     )
        #     dec_target_metrics_every[str(iii)] += list(
        #         dec_target[:, iii, :].view(-1, num_class).cpu().numpy()
        #     )
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

    path = "/home/aleflabo/ego_procedural/OadTR/models/onlyThis_allMistakes_en_3_decoder_5_lr_drop_1_512/results"  # /home/aleflabo/ego_procedural/OadTR/models/onlyThis_train+val_en_3_decoder_5_lr_drop_1_512/results'
    with open(f"{path}/video_names_test.pickle", "wb") as outfile:
        pickle.dump(video_names, outfile)
    with open(f"{path}/start_frame.pickle", "wb") as outfile:
        pickle.dump(start_frame, outfile)
    with open(f"{path}/end_frame.pickle", "wb") as outfile:
        pickle.dump(end_frame, outfile)
    with open(f"{path}/results_enc_test.pickle", "wb") as outfile:
        pickle.dump(all_probs, outfile)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # accuracy, precision, recall, specificity, f1_score = metrics_calculator.statistics_epoch()
    # print('average accuracy:', accuracy,'average precision:', precision, 'average recall:',recall, 'average specificity:',specificity, 'average f1 score:',f1_score)

    # results
    # ?all_probs = np.asarray(all_probs).T
    #     logger.output_print(str(all_probs.shape))  # (21, 180489)

    # ?all_classes = np.asarray(all_classes).T
    #     logger.output_print(str(all_classes.shape))  # (21, 180489)
    # ?results = {'probs': all_probs, 'labels': all_classes}

    #     map, aps, _, _ = utils.frame_level_map_n_cap(results)
    #     logger.output_print('[Epoch-{}] [IDU-{}] mAP: {:.4f}\n'.format(epoch, feat_type, map))

    #     wandb.log({'mAP': map})

    # results_dec = {}
    # results_dec['probs'] = np.asarray(dec_score_metrics).T
    # results_dec['labels'] = np.asarray(dec_target_metrics).T
    # with open(
    #     "data_test/encoder_only-baseline/results_enc_test.pickle", "wb"
    # ) as outfile:
    #     pickle.dump(results, outfile)
    # with open('/home/aleflabo/ego_procedural/OadTR/models/en_3_decoder_5_lr_drop_1_train/results/results_dec_test.pickle', 'wb') as outfile:
    #     pickle.dump(results_dec, outfile)

    # dec_map_2, dec_aps_2, _, _ = util.frame_level_map_n_cap_thumos(results_dec)
    # logger.output_print('dec_mAP all together: | {} |.'.format(dec_map_2))

    # !Commented for now: Decoder
    # all_decoder = 0.
    # for iii in range(num_query):
    #     results_dec = {}
    #     results_dec['probs'] = np.asarray(dec_score_metrics_every[str(iii)]).T
    #     results_dec['labels'] = np.asarray(dec_target_metrics_every[str(iii)]).T

    #     # with open('/home/aleflabo/ego_procedural/OadTR/models/en_3_decoder_5_lr_drop_1/results/results_dec_test_{}.pickle'.format(iii), 'wb') as outfile:
    #     #     pickle.dump(results_dec, outfile)

    #     dec_map_2, dec_aps_2, _, _ = util.frame_level_map_n_cap_thumos(results_dec)
    #     logger.output_print('dec_mAP_pred | {} : {} |.'.format(iii, dec_map_2))
    #     all_decoder += dec_map_2
    # logger.output_print('{}: | {:.4f} |.'.format('all decoder map', all_decoder/num_query))

    #     for i, ap in enumerate(aps):
    #         cls_name = all_class_name[i]
    #         logger.output_print('{}: {:.4f}'.format(cls_name, ap))
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # ? with open ('data_test/encoder_only-baseline/start_frame.pickle', 'wb') as outfile:
    # ?    pickle.dump(start_frame, outfile)
    # ? with open ('data_test/encoder_only-baseline/end_frame.pickle', 'wb') as outfile:
    # ?    pickle.dump(end_frame, outfile)

    (
        accuracy_epoch,
        precision_epoch,
        recall_epoch,
        f1_score_epoch,
        auc_roc_epoch,
        MAP_epoch,
    ) = metrics_calculator.statistics_epoch()
    print(
        "accuracy epoch:",
        accuracy_epoch,
        "precision epoch:",
        precision_epoch,
        "recall epoch:",
        recall_epoch,
        "f1_score epoch:",
        f1_score_epoch,
        "auc_roc epoch:",
        auc_roc_epoch,
        "MAP epoch:",
        MAP_epoch,
    )

    # return #stats
    return
