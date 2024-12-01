import torch
import torch.nn as nn
from tqdm import tqdm
import time
from utils import thumos_postprocessing
from utils import *
import json
from trainer.eval_builder import EVAL
from utils import thumos_postprocessing, perframe_average_precision
import pickle
import numpy as np
import os


@EVAL.register("OAD")
class Evaluate(nn.Module):

    def __init__(self, cfg):
        super(Evaluate, self).__init__()
        self.data_processing = (
            thumos_postprocessing if "THUMOS" in cfg["data_name"] else None
        )
        self.metric = cfg["metric"]
        self.eval_method = perframe_average_precision
        self.cfg = cfg
        self.all_class_names = json.load(open(cfg["video_list_path"]))[
            cfg["data_name"].split("_")[0]
        ]["class_index"]

    def eval(self, model, dataloader, logger, device):
        model.eval()
        output = {}
        with torch.no_grad():
            pred_scores, gt_targets = [], []
            start = time.time()
            for rgb_input, flow_input, target, vid, start, end in tqdm(
                dataloader, desc="Evaluation:", leave=False
            ): 
                rgb_input, flow_input, target = (
                    rgb_input.to(device),
                    flow_input.to(device),
                    target.to(device),
                )
                out_dict = model(rgb_input, flow_input)
                pred_logit = out_dict["logits"]
                prob_val = pred_logit.squeeze().cpu().numpy()
                target_batch = target.squeeze().cpu().numpy()
                pred_scores += list(prob_val)
                gt_targets += list(target_batch)
                #! To save the prediction scores and ground truth
                if self.cfg['eval'] != None:
                    video_name = vid[0]
                    pred = np.argmax(prob_val, axis=1)
                    gt = np.argmax(target_batch, axis=1)
                    sample = {"pred": pred, "gt": gt}
                    output[video_name] = sample

            # save output as json file
            if self.cfg['eval'] != None:
                os.makedirs("output_miniRoad", exist_ok=True)
                # save as json file
                for k, v in output.items():
                    output[k] = {"pred": v["pred"].tolist(), "gt": v["gt"].tolist()}    
                with open("output_miniRoad/output_miniROAD.json", "w") as file:
                    json.dump(output, file)


            end = time.time()
            num_frames = len(gt_targets)
            result = self.eval_method(
                pred_scores,
                gt_targets,
                self.all_class_names,
                self.data_processing,
                self.metric,
            )
            time_taken = (end - start).item()
            logger.info(
                f"Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)"
            )
        return result["mean_AP"]

    def forward(self, model, dataloader, logger, device):
        return self.eval(model, dataloader, logger, device)


@EVAL.register("ANTICIPATION")
class ANT_Evaluate(nn.Module):

    def __init__(self, cfg):
        super(ANT_Evaluate, self).__init__()
        data_name = cfg["data_name"].split("_")[0]
        self.data_processing = thumos_postprocessing if data_name == "THUMOS" else None
        self.metric = cfg["metric"]
        self.eval_method = perframe_average_precision
        self.all_class_names = json.load(open(cfg["video_list_path"]))[data_name][
            "class_index"
        ]

    def eval(self, model, dataloader, logger):
        device = "cuda:0"
        model.eval()
        with torch.no_grad():
            pred_scores, gt_targets, ant_pred_scores, ant_gt_targets = [], [], [], []
            start = time.time()
            anticipation_mAPs = []
            for rgb_input, flow_input, target, ant_target in tqdm(
                dataloader, desc="Evaluation:", leave=False
            ):
                rgb_input, flow_input, target, ant_target = (
                    rgb_input.to(device),
                    flow_input.to(device),
                    target.to(device),
                    ant_target.to(device),
                )
                out_dict = model(rgb_input, flow_input)
                pred_logit = out_dict["logits"]
                ant_pred_logit = out_dict["anticipation_logits"]
                prob_val = pred_logit.squeeze().cpu().numpy()
                target_batch = target.squeeze().cpu().numpy()
                ant_prob_val = ant_pred_logit.squeeze().cpu().numpy()
                ant_target_batch = ant_target.squeeze().cpu().numpy()
                pred_scores += list(prob_val)
                gt_targets += list(target_batch)
                ant_pred_scores += list(ant_prob_val)
                ant_gt_targets += list(ant_target_batch)
            end = time.time()
            num_frames = len(gt_targets)
            result = self.eval_method(
                pred_scores,
                gt_targets,
                self.all_class_names,
                self.data_processing,
                self.metric,
            )
            ant_pred_scores = np.array(ant_pred_scores)
            ant_gt_targets = np.array(ant_gt_targets)
            logger.info(f'OAD mAP: {result["mean_AP"]*100:.2f}')
            for step in range(ant_gt_targets.shape[1]):
                result[f"anticipation_{step+1}"] = self.eval_method(
                    ant_pred_scores[:, step, :],
                    ant_gt_targets[:, step, :],
                    self.all_class_names,
                    self.data_processing,
                    self.metric,
                )
                anticipation_mAPs.append(result[f"anticipation_{step+1}"]["mean_AP"])
                logger.info(
                    f"Anticipation at step {step+1}: {result[f'anticipation_{step+1}']['mean_AP']*100:.2f}"
                )
            logger.info(f"Mean Anticipation mAP: {np.mean(anticipation_mAPs)*100:.2f}")

            time_taken = end - start
            logger.info(
                f"Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)"
            )

        return np.mean(anticipation_mAPs)

    def forward(self, model, dataloader, logger):
        return self.eval(model, dataloader, logger)
