from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class LM(nn.Module):
    def __init__(self, mask_mode: str, lm: str, p: Optional[float]) -> None:
        """
        Args:
            mask_mode (str): Mask mode. Either prob or end
            lm (str): Language model. Either bert or gpt2
            p (Optional[float]): Probability of masking
        """
        super().__init__()

        self.backbone, self.tokenizer = None, None

        if mask_mode == "prob":
            assert p is not None, "p should be specified"

        self.p = p
        self.mask_mode = mask_mode

        if lm == "bert":
            self.backbone = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif lm == "gpt2":
            self.backbone = AutoModelForMaskedLM.from_pretrained("gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            raise ValueError("lm should be either bert or gpt2")

        # * freeze all the parameters except for cls.predictions
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.cls.predictions.parameters():
            param.requires_grad = True

    def forward(self, hist: dict) -> tuple:
        """
        Args:
            hist (dict): History
        Returns:
            out (torch.Tensor): Output logits
        """
        input_ids, attention_mask, labels = self.tokenize(self.mask_mode, hist, self.p)
        input_ids = input_ids.to(device=self.backbone.device)
        attention_mask = attention_mask.to(device=self.backbone.device)
        labels = labels.to(device=self.backbone.device)

        out = self.backbone(
            input_ids, attention_mask=attention_mask, labels=labels
        ).logits
        return out, input_ids

    def tokenize(self, mode: str, x: list, p: Optional[float] = None):
        """
        Tokenize wrapper.

        Args:
            mode (str): Tokenize mode. Either prob or end
            x (list): List of tokens
            p (Optional[float]): Probability of masking

        Returns:
            input_ids (torch.Tensor): Input ids
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor): Labels
        """
        if mode == "end":
            return self.tokenize_end(x)
        elif mode == "prob":
            assert p is not None, "p should be specified"
            return self.tokenize_prob(x, p)
        elif mode == "none":
            return self.tokenize_none(x)

    def tokenize_end(self, x: list) -> tuple:
        """
        Tokenize adding [MASK] at the end of the sentence three times.

        Args:
            x (list): List of tokens

        Returns:
            input_ids (torch.Tensor): Input ids
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor): Labels
        """
        texts = []
        for entry in x:
            text = " ".join(entry)
            text = text + 3 * " [MASK]"
            texts.append(text)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask, input_ids

    def tokenize_prob(self, x: list, p: float) -> tuple:
        """
        Tokenize adding [MASK] with probability p.

        Args:
            x (list): List of tokens
            p (float): Probability of masking

        Returns:
            input_ids (torch.Tensor): Input ids
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor): Labels
        """
        input_ids, attention_mask, labels = self.tokenize_none(x)
        labels = labels.detach().clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (
            (rand < p) * (input_ids != 101) * (input_ids != 102) * (input_ids != 0)
        )

        selection = []
        for i in range(input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
        for i in range(input_ids.shape[0]):
            input_ids[i, selection[i]] = 103

        return input_ids, attention_mask, labels

    def tokenize_none(self, x: list) -> tuple:
        """
        Tokenize without adding [MASK].

        Args:
            x (list): List of tokens

        Returns:
            input_ids (torch.Tensor): Input ids
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor): Labels
        """
        texts = []
        for entry in x:
            text = " ".join(entry)
            texts.append(text)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask, input_ids
