import numpy as np
import torch
from tqdm import tqdm

from src.Graph2Vid.dp.exact_dp import crosstask_dp


def framewise_accuracy(frame_assignment, sample, use_unlabeled=False):
    """calculate framewise accuracy as done in COIN"""
    num_steps = sample["num_steps"]
    num_frames = sample["num_frames"]
    if isinstance(num_steps, (np.ndarray, torch.Tensor)):
        num_steps, num_frames = num_steps.item(), num_frames.item()

    # non-step frames/clips are assigned label = -1
    gt_assignment = -np.ones(num_frames, dtype=np.int32)
    # convert start and end times to clip/frame -wise labels
    for s in np.arange(num_steps):
        st_ed = np.arange(sample["step_starts"][s], sample["step_ends"][s] + 1)
        gt_assignment[st_ed] = s

    # to discount unlabeled frames in gt
    if not use_unlabeled:
        unlabled = np.count_nonzero(gt_assignment == -1)
        num_frames = num_frames - unlabled
        fa = np.logical_and(
            frame_assignment == gt_assignment, gt_assignment != -1
        ).sum()
    else:
        fa = np.count_nonzero((frame_assignment == gt_assignment))
    # framewise accuracy
    fa = fa / num_frames if num_frames != 0 else 0
    return fa


def IoU(frame_assignment, sample):
    """calculate IoU as done in COIN"""
    num_steps = sample["num_steps"]
    num_frames = sample["num_frames"]
    if isinstance(num_steps, (np.ndarray, torch.Tensor)):
        num_steps, num_frames = num_steps.item(), num_frames.item()

    # non-step frames/clips are assigned label = -1
    gt_assignment = -np.ones((num_frames,), dtype=np.int32)
    # convert start and end times to clip/frame -wise labels
    intersection, union = 0, 0
    for s in range(num_steps):
        st_ed = np.arange(sample["step_starts"][s], sample["step_ends"][s] + 1)
        gt_assignment[st_ed] = s
        intersection += np.logical_and(gt_assignment == s, frame_assignment == s).sum()
        union += np.logical_or(gt_assignment == s, frame_assignment == s).sum()
    return intersection / union


def IoU_class(frame_assignment, gt_assignment, sample=None):
    """calculate IoU as done in COIN"""

    if not isinstance(gt_assignment, np.ndarray):
        gt_assignment = gt_assignment.numpy()
    else:
        gt_assignment = np.array(gt_assignment)
    if not isinstance(frame_assignment, np.ndarray):
        frame_assignment = frame_assignment.numpy()
    else:
        frame_assignment = np.array(frame_assignment)

    if sample is not None:
        # color frame assignment with classes
        for i, cls_id in enumerate(sample["step_ids"]):
            frame_assignment[frame_assignment == i] = cls_id

    present_classes = np.unique(gt_assignment)
    present_classes = present_classes[present_classes > -1]
    intersection, union = 0, 0
    for s, cls_id in enumerate(present_classes):
        gt_cls_seg = gt_assignment == cls_id
        pred_cls_seg = frame_assignment == cls_id
        intersection += np.logical_and(gt_cls_seg, pred_cls_seg).sum()
        union += np.logical_or(gt_cls_seg, pred_cls_seg).sum()
    return intersection / union


def Acc_class(frame_assignment, gt_assignment, sample=None, use_negative=True):
    """calculate IoU as done in COIN"""
    # color frame assignment with classes
    if not isinstance(gt_assignment, np.ndarray):
        gt_assignment = gt_assignment.numpy()
    else:
        gt_assignment = np.array(gt_assignment)
    if not isinstance(frame_assignment, np.ndarray):
        frame_assignment = frame_assignment.numpy()
    else:
        frame_assignment = np.array(frame_assignment)

    if sample is not None:
        for i, cls_id in enumerate(sample["step_ids"]):
            frame_assignment[frame_assignment == i] = cls_id

    if not use_negative:
        frame_assignment = frame_assignment[gt_assignment > -1]
        gt_assignment = gt_assignment[gt_assignment > -1]
    return (frame_assignment == gt_assignment).astype(float).mean()


def recall_crosstask(val_loader, model):
    "Recall as defined in CrossTask"
    total_steps = 0
    detected_steps_clips = 0
    detected_steps_seconds = 0
    for i, sample in enumerate(tqdm(val_loader)):
        if sample["num_steps"] < 1:
            continue

        device = (
            "cuda"
            if (model is not None and next(model.parameters()).is_cuda)
            else "cpu"
        )
        if model is not None:
            frame_features = (
                model.map_video(sample["frame_features"].to(device)).detach().cpu()
            )
            step_features = (
                model.map_text(sample["step_features"].to(device)).detach().cpu()
            )
        else:
            frame_features = sample["frame_features"].cpu()
            step_features = sample["step_features"].cpu()
        text_clip_similarity = (step_features @ frame_features.T).detach().cpu().numpy()

        optimal_assignment = crosstask_dp(-text_clip_similarity.T).argmax(0)
        for si in range(sample["num_steps"]):
            # eval on seconds
            start, end = sample["step_starts_sec"][si], sample["step_ends_sec"][si]
            infer_t = (optimal_assignment[si] + 0.5) * 3.2
            detected_steps_seconds += int(infer_t >= start and infer_t <= end)

            # eval on clips level
            start, end = sample["step_starts"][si], sample["step_ends"][si]
            infer_t = optimal_assignment[si]
            detected_steps_clips += int(infer_t >= start and infer_t <= end)

            total_steps += 1

    sec_recall = 100 * detected_steps_seconds / total_steps
    return sec_recall
