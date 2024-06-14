from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn


class Splitter(nn.Module):
    def __init__(self, in_dim: int, out_dims: Dict) -> None:
        super().__init__()

        self.fc_verb = nn.Sequential(nn.Linear(in_dim, out_dims["verb"]), nn.Sigmoid())
        self.fc_this = nn.Linear(in_dim + out_dims["verb"], out_dims["this"])
        self.fc_that = nn.Linear(in_dim + out_dims["this"], out_dims["that"])

    def forward(self, x: torch.Tensor) -> Dict:
        out = {}
        out["verb"] = self.fc_verb(x)
        out["this"] = self.fc_this(torch.cat([x, out["verb"]], dim=2))
        out["that"] = self.fc_that(torch.cat([x, out["this"]], dim=2))
        return out


class Model(nn.Module):
    def __init__(
        self, in_dim: int, h_dim: int, num_layers: int, batch_first: bool = True
    ) -> None:
        super().__init__()
        # TODO separate the RNNs for verb, this and that.
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.splitter = Splitter(
            in_dim=h_dim, out_dims={"verb": 1, "this": in_dim - 2, "that": in_dim - 2}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, (h, c) = self.rnn(x)
        out = self.splitter(out)

        return out
