from ast import mod

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.data.assembly_text import AssemblyTextDataset
from src.models.lm import LM
from src.utils.parser import args

if __name__ == "__main__":
    device = torch.device(args.device)
    wandb.init(
        name=args.wandb_name,
        group=args.wandb_group,
        project="egoprocel",
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
    )
    # * Dataset
    train_ds = AssemblyTextDataset("data/mistake_labels", split="train")
    val_ds = AssemblyTextDataset("data/mistake_labels", split="test")
    # * Dataloader
    # TODO don't know why but batch size higher than 2 gives problem to tokenization
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=train_ds.collate_fn
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=val_ds.collate_fn
    )

    # * Model
    model = LM(lm=args.lm, p=args.tokenize_prob, mask_mode=args.mask_mode)
    model.to(device=device)

    # * Optim
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # * Validation
    # TODO perplexity
    # TODO cosine similarity

    for epoch in tqdm(range(args.epochs)):
        model.train()
        loss_tot = 0
        print(f"Epoch {epoch}")
        for batch in train_dl:
            hist = batch["hist"]

            out, input_ids = model(hist)
            # get indices of [MASK]
            mask_token_index = torch.where(input_ids == model.tokenizer.mask_token_id)
            if len(mask_token_index[0]) == 0:
                continue

            if args.mask_mode == "prob":
                y, _, _ = model.tokenize(x=hist, mode="none")
                y = y[(mask_token_index[0].cpu(), mask_token_index[1].cpu())]
            elif args.mask_mode == "end":
                gt = batch["gt"]
                y = model.tokenizer(gt, return_tensors="pt", padding=True).input_ids
                # equivalent, get the last three masks
                y = y[:, 1:4].flatten()
                # out = out[:, -4:-1, :]
            else:
                raise ValueError("mask_mode should be either prob or end")

            loss = loss_fn(out[mask_token_index].cpu(), y)
            loss.backward()
            optim.step()

            loss_tot += loss.item()

            if args.debug:
                break

        loss = loss_tot / len(train_dl)
        print(f"\tloss {loss}")
        wandb.log({"CEloss": loss})

        # * Validation
        if epoch % args.validate_every == 0:
            model.eval()

            acc_tot, acc_len = 0, 0
            nll = 0
            for batch in val_dl:
                hist = batch["hist"]

                out, input_ids = model(hist)
                # get indices of [MASK]
                mask_token_index = torch.where(
                    input_ids == model.tokenizer.mask_token_id
                )

                if args.mask_mode == "prob":
                    y, _, _ = model.tokenize(x=hist, mode="none")
                    y = y[(mask_token_index[0].cpu(), mask_token_index[1].cpu())]
                elif args.mask_mode == "end":
                    gt = batch["gt"]
                    y = model.tokenizer(gt, return_tensors="pt", padding=True).input_ids
                    # equivalent, get the last three masks, y has CLS and SEP
                    y = y[:, 1:4].flatten()
                else:
                    raise ValueError("mask_mode should be either prob or end")

                # Accuracy
                pred = torch.argmax(out, dim=-1)
                acc_tot += torch.sum(pred[mask_token_index].cpu() == y)
                acc_len += len(mask_token_index[0])

                if args.debug:
                    break

            acc = acc_tot / acc_len
            print(f"\tacc {acc}")
            wandb.log({"acc": acc})
        ####################################################################
        # # * BERT
        # texts = []
        # for entry in hist:
        #     text = " ".join(entry)
        #     text = text + 3 * " [MASK]"
        #     texts.append(text)

        # # TODO maybe we should make it autoregressive to avoid predicting always the same word
        # inputs = tokenizer(texts, return_tensors="pt")
        # token_logits = model(**inputs).logits
        # # Find the location of [MASK] and extract its logits
        # mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)

        # mask_token_logits = token_logits[mask_token_index[0], mask_token_index[1], :]
        # N, D = mask_token_logits.shape
        # mask_token_logits = mask_token_logits.reshape(B, N // B, D)
        # # Pick the [MASK] candidates with the highest logits
        # top_k_tokens = torch.topk(mask_token_logits, 1, dim=-1).indices.transpose(1, 2)

        # for i, (text, k_tokens) in enumerate(zip(texts, top_k_tokens)):
        #     print(f"{i}: text")
        #     for tokens in k_tokens:
        #         print(
        #             f'{i}: {text.replace("[MASK] [MASK] [MASK]", tokenizer.decode(tokens))}'
        #         )

        if args.debug:
            break
