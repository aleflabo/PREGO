import os
from time import time
from typing import Dict, Tuple

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

import wandb
from src.data.assemblyLabelDataset import AssemblyLabelDataset
from src.data.dataset_utils import collate_fn
from src.models.lstm import Model as LSTM
from src.utils.metrics import MultiHotAccuracy
from src.utils.parser import args


def get_windows(
    x: torch.Tensor, in_window_len: int, out_window_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split the input tensor in windows of size in_window_len and return the input and ground truth tensors.

    Args:
        x (torch.Tensor): input tensor of shape [B, S, K].
        in_window_len (int): length of the input window.
        out_window_len (int): length of the output window.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: input and ground truth tensors.
    """
    assert (
        in_window_len < x.shape[1]
    ), "in_window_len must be less than the sequence length"

    num_windows = x.shape[1] // in_window_len
    x = x[:, : num_windows * in_window_len, :].view(
        x.size(0), num_windows, in_window_len, x.size(2)
    )

    # ground truth
    y = x[:, 1:, :, :]
    if out_window_len < in_window_len:
        y = y[:, :out_window_len, :, :]
    elif out_window_len > in_window_len:
        # TODO: we could do it taking values from the next sequences until we reach the desired length
        raise ValueError("out_window_len must be less than in_window_len")

    # input
    x = x[:, :-1, :, :]

    return x, y


def get_gt_preds(
    gt: torch.Tensor, preds: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Get ground truth and predictions for the verb, this and that; formatting them correctly to be compared.

    Args:
        gt (torch.Tensor): ground truth tensor of shape [B, K, S, D].
        preds (Dict[str, torch.Tensor]): dictionary of predictions of shape [B, K, S, D].

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: dictionary of ground truth and predictions.
    """
    out_window = gt.shape[1]
    # * We optimize on the verb "attach".
    gt_verb = gt[:, :, :, :1]
    pred_verb = preds["verb"][:, :out_window, :, :]

    # * optimizer for this and that separately
    gt_parts = gt[:, :, :, 2:]

    # get indices of ones or of two
    gt_this = torch.zeros(
        (gt_parts.shape[0], gt_parts.shape[1], gt_parts.shape[2], 1), dtype=torch.long
    )
    gt_that = torch.zeros(
        (gt_parts.shape[0], gt_parts.shape[1], gt_parts.shape[2], 1), dtype=torch.long
    )

    # * We have to deal separately with the case in which we have two parts and the case
    # * in which we have one part assembled with itself.
    # this != that
    idxs = torch.stack(torch.where(gt_parts == 1))
    idxs_this, idxs_that = idxs[:, ::2], idxs[:, 1::2]

    # assign the index to the corresponding entry
    gt_this[idxs_this[0], idxs_this[1], idxs_this[2]] = idxs_this[-1].unsqueeze(dim=-1)
    gt_that[idxs_that[0], idxs_that[1], idxs_that[2]] = idxs_that[-1].unsqueeze(dim=-1)

    # this == that
    idxs = torch.stack(torch.where(gt_parts == 2))
    gt_this[idxs[0], idxs[1], idxs[2]] = idxs[-1].unsqueeze(dim=-1)
    gt_that[idxs[0], idxs[1], idxs[2]] = idxs[-1].unsqueeze(dim=-1)

    pred_this = preds["this"][:, :out_window, :, :]
    pred_that = preds["that"][:, :out_window, :, :]

    return {
        "gt": {"verb": gt_verb, "this": gt_this, "that": gt_that},
        "pred": {"verb": pred_verb, "this": pred_this, "that": pred_that},
    }


def get_loss(
    gt: torch.Tensor,
    preds: Dict[str, torch.Tensor],
    loss_fn: Dict[str, nn.Module],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the loss for the given ground truth and predictions.

    Args:
        gt (torch.Tensor): ground truth tensor of shape [B, K, S, D].
        preds (Dict[str, torch.Tensor]): dictionary of predictions of shape [B, K, S, D].
        loss_fn (Dict[str, nn.Module]): dictionary of loss functions.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: loss and dictionary of losses.
    """
    d = get_gt_preds(gt=gt, preds=preds)
    gt_verb, gt_this, gt_that = d["gt"]["verb"], d["gt"]["this"], d["gt"]["that"]
    pred_verb, pred_this, pred_that = (
        d["pred"]["verb"],
        d["pred"]["this"],
        d["pred"]["that"],
    )

    # Compute loss
    loss_verb = loss_fn["verb"](pred_verb, gt_verb)
    loss_this = loss_fn["this"](pred_this.permute(0, 3, 1, 2), gt_this.squeeze(dim=-1))
    loss_that = loss_fn["that"](pred_that.permute(0, 3, 1, 2), gt_that.squeeze(dim=-1))

    loss = loss_verb + loss_this + loss_that
    return loss, {"verb": loss_verb, "this": loss_this, "that": loss_that}


def get_metrics(
    gt: torch.Tensor,
    preds: Dict[str, torch.Tensor],
    metrics: Dict,
):
    """
    Get the metrics for the given ground truth and predictions.

    Args:
        gt (torch.Tensor): ground truth tensor of shape [B, K, S, D].
        preds (Dict[str, torch.Tensor]): dictionary of predictions of shape [B, K, S, D].
    """
    d = get_gt_preds(gt=gt, preds=preds)
    gt_verb, gt_this, gt_that = d["gt"]["verb"], d["gt"]["this"], d["gt"]["that"]
    pred_verb, pred_this, pred_that = (
        d["pred"]["verb"],
        d["pred"]["this"],
        d["pred"]["that"],
    )

    # * Metrics
    # verb
    # threshold = 0.5 and convert to int
    gt_verb = gt_verb.reshape(-1)
    pred_verb = pred_verb.reshape(-1)
    metrics["verb"]["accuracy"].update(pred_verb, gt_verb)
    metrics["verb"]["precision"].update(pred_verb, gt_verb)
    metrics["verb"]["recall"].update(pred_verb.to(torch.uint8), gt_verb.to(torch.uint8))
    metrics["verb"]["f1"].update(pred_verb, gt_verb)

    # this
    gt_this = gt_this.reshape(-1)
    pred_this = pred_this.argmax(dim=-1).reshape(-1)
    metrics["this"]["accuracy"].update(pred_this, gt_this)
    metrics["this"]["precision"].update(pred_this, gt_this)
    metrics["this"]["recall"].update(pred_this, gt_this)
    metrics["this"]["f1"].update(pred_this, gt_this)

    # that
    gt_that = gt_that.reshape(-1)
    pred_that = pred_that.argmax(dim=-1).reshape(-1)
    metrics["that"]["accuracy"].update(pred_that, gt_that)
    metrics["that"]["precision"].update(pred_that, gt_that)
    metrics["that"]["recall"].update(pred_that, gt_that)
    metrics["that"]["f1"].update(pred_that, gt_that)

    # all
    pred = preds["all"]
    metrics["all"]["accuracy"].update(preds=pred, target=gt)


def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: Dict):
    """
    Test the model on the given dataloader.

    Args:
        dataloader (DataLoader): dataloader.
        model (nn.Module): model.
        loss_fn (Dict): dictionary of loss functions.
    """
    model.eval()

    # Metrics
    metrics = {
        "verb": {
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
        },
        "this": {
            "accuracy": MulticlassAccuracy(),
            "precision": MulticlassPrecision(),
            "recall": MulticlassRecall(),
            "f1": MulticlassF1Score(),
        },
        "that": {
            "accuracy": MulticlassAccuracy(),
            "precision": MulticlassPrecision(),
            "recall": MulticlassRecall(),
            "f1": MulticlassF1Score(),
        },
        "all": {
            "accuracy": MultiHotAccuracy(),
        },
    }

    with torch.no_grad():
        for batch in dataloader:
            sample = batch["oh_sample"]

            # Divide the sample in windows
            x, gt = get_windows(
                sample,
                in_window_len=cfg.train.in_seq_len,
                out_window_len=cfg.train.out_seq_len,
            )

            verbs_l, this_l, that_l = [], [], []
            preds_l = []
            for i in range(x.shape[1]):
                pred = model(x[:, i, ...])

                verbs_l.append(pred["verb"])
                this_l.append(pred["this"])
                that_l.append(pred["that"])

                preds_l.append(
                    torch.cat(
                        (pred["verb"], 1 - pred["verb"], pred["this"] + pred["that"]),
                        dim=-1,
                    )
                )

            preds = {
                "verb": torch.stack(verbs_l, dim=1),
                "this": torch.stack(this_l, dim=1),
                "that": torch.stack(that_l, dim=1),
                "all": torch.stack(preds_l, dim=1),
            }

            get_metrics(preds=preds, gt=gt, metrics=metrics)

    verb_acc = metrics["verb"]["accuracy"].compute()
    verb_prec = metrics["verb"]["precision"].compute()
    verb_rec = metrics["verb"]["recall"].compute()
    verb_f1 = metrics["verb"]["f1"].compute()

    this_acc = metrics["this"]["accuracy"].compute()
    this_prec = metrics["this"]["precision"].compute()
    this_rec = metrics["this"]["recall"].compute()
    this_f1 = metrics["this"]["f1"].compute()

    that_acc = metrics["that"]["accuracy"].compute()
    that_prec = metrics["that"]["precision"].compute()
    that_rec = metrics["that"]["recall"].compute()
    that_f1 = metrics["that"]["f1"].compute()

    # all
    acc = metrics["all"]["accuracy"].compute()

    print(
        f"[Test] verb: {verb_acc:>8f} \t this: {this_acc:>8f} \t that: {that_acc:>8f}"
    )

    wandb.log(
        {
            "test/verb_acc": verb_acc,
            "test/verb_prec": verb_prec,
            "test/verb_rec": verb_rec,
            "test/verb_f1": verb_f1,
            "test/this_acc": this_acc,
            "test/this_prec": this_prec,
            "test/this_rec": this_rec,
            "test/this_f1": this_f1,
            "test/that_acc": that_acc,
            "test/that_prec": that_prec,
            "test/that_rec": that_rec,
            "test/that_f1": that_f1,
            "test/acc": acc,
        }
    )


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Dict,
    in_seq_len: int = 1,
    out_seq_len: int = 1,
):
    """
    Train the model on the given dataloader.

    Args:
        dataloader (DataLoader): dataloader.
        model (nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.
        loss_fn (Dict): dictionary of loss functions.
    """
    model.train()

    loss_tot = 0
    loss_d_tot = {"verb": 0, "this": 0, "that": 0}
    for batch_idx, batch in enumerate(dataloader):
        sample = batch["oh_sample"]  # [B,K,S]

        # Divide the sample in windows
        x, gt = get_windows(
            sample, in_window_len=in_seq_len, out_window_len=out_seq_len
        )

        verbs_l, this_l, that_l = [], [], []
        preds_l = []
        for i in range(x.shape[1]):
            pred = model(x[:, i, ...])

            verbs_l.append(pred["verb"])
            this_l.append(pred["this"])
            that_l.append(pred["that"])

            preds_l.append(
                torch.cat(
                    (pred["verb"], 1 - pred["verb"], pred["this"] + pred["that"]),
                    dim=-1,
                )
            )

        preds = {
            "verb": torch.stack(verbs_l, dim=1),
            "this": torch.stack(this_l, dim=1),
            "that": torch.stack(that_l, dim=1),
            "all": torch.stack(preds_l, dim=1),
        }

        loss, loss_d = get_loss(preds=preds, gt=gt, loss_fn=loss_fn)
        loss_tot += loss.item()
        loss_d_tot["verb"] += loss_d["verb"].item()
        loss_d_tot["this"] += loss_d["this"].item()
        loss_d_tot["that"] += loss_d["that"].item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log
        if batch_idx % 10 == 0:
            loss, loss_verb, loss_this, loss_that, current = (
                loss.item(),
                loss_d["verb"].item(),
                loss_d["this"].item(),
                loss_d["that"].item(),
                (batch_idx + 1) * len(sample),
            )

            # loss, current = loss.item(), (batch_idx + 1) * len(sample)
            print(
                f"loss: {loss:>7f} \t verb: {loss_verb:>7f} \t this: {loss_this:>7f} \t that: {loss_that:>7f} [{current:>5d}/{len(dataloader.dataset):>5d}]"
            )

        if args.debug:
            break

    data_len = len(dataloader.dataset)
    loss_tot /= data_len
    loss_d_tot["verb"] /= data_len
    loss_d_tot["this"] /= data_len
    loss_d_tot["that"] /= data_len

    print(
        f"[Train] avg: {loss_tot:>8f} \t verb: {loss_d_tot['verb']:>8f} \t this: {loss_d_tot['this']:>8f} \t that: {loss_d_tot['that']:>8f}"
    )

    wandb.log(
        {
            "train/loss": loss_tot,
            "train/loss_verb": loss_d_tot["verb"],
            "train/loss_this": loss_d_tot["this"],
            "train/loss_that": loss_d_tot["that"],
        }
    )

    if cfg.log.save_interval > 0 and epoch % cfg.log.save_interval == 0:
        torch.save(
            model.state_dict(),
            cfg.log.path + args.wandb_name + "/ckpt/" + str(epoch) + ".pth",
        )


if __name__ == "__main__":
    # * Setup
    global cfg
    cfg = OmegaConf.load(args.cfg)

    # Log
    if not os.path.exists(cfg.log.path + args.wandb_name):
        os.makedirs(cfg.log.path + args.wandb_name)
        os.makedirs(cfg.log.path + args.wandb_name + "/ckpt")

    wandb.init(
        project=cfg.log.wandb.project,
        entity=cfg.log.wandb.entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        name=args.wandb_name,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
        config={**vars(args), **vars(cfg)},
    )

    # * Data
    path_to_csv_A101 = cfg.data.path
    train_dataset = AssemblyLabelDataset(path_to_csv_A101, split="correct")
    test_dataset = AssemblyLabelDataset(path_to_csv_A101, split="mistake")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=cfg.test.shuffle,
        collate_fn=collate_fn,
    )

    # * Model
    model = LSTM(
        in_dim=cfg.model.in_dim,
        h_dim=cfg.model.h_dim,
        num_layers=cfg.model.num_layers,
        batch_first=cfg.model.batch_first,
    )

    # * Optimization
    loss_fn = {
        "verb": nn.BCELoss(),
        "this": nn.CrossEntropyLoss(),
        "that": nn.CrossEntropyLoss(),
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optim.learning_rate)

    # * Train
    s = time()
    for epoch in range(cfg.train.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        wandb.log({"epoch": epoch + 1})
        train_loop(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            in_seq_len=cfg.train.in_seq_len,
            out_seq_len=cfg.train.out_seq_len,
        )

        if epoch % cfg.log.log_interval == 0:
            test_loop(dataloader=test_dataloader, model=model, loss_fn=loss_fn)
    wandb.log({"time": time() - s})
