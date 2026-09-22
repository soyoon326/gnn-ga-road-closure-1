from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


_LEGACY_EDGE_MLP_REMAP = {"edge_mlp.2.": "edge_mlp.3.", "edge_mlp.4.": "edge_mlp.6."}


def remap_legacy_state_dict(state_dict, reference):
    out = {}
    for key, value in state_dict.items():
        for old, new in _LEGACY_EDGE_MLP_REMAP.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        out[key] = value
    if set(out) != set(reference):
        return None
    if any(out[k].shape != reference[k].shape for k in reference):
        return None
    return out


class PolicyGNN(nn.Module):

    def __init__(
        self,
        in_node: int,
        in_edge: int,
        hid: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        if hid % heads != 0:
            raise ValueError(f"hid must be divisible by heads: hid={hid}, heads={heads}")
        self.dropout_p = float(dropout)
        self.use_layernorm = bool(use_layernorm)

        self.n_lin = nn.Linear(in_node, hid)
        self.gat1 = GATv2Conv(hid, hid // heads, heads=heads, add_self_loops=True, edge_dim=in_edge)
        self.gat2 = GATv2Conv(hid, hid // heads, heads=heads, add_self_loops=True, edge_dim=in_edge)
        self.ln0 = nn.LayerNorm(hid) if self.use_layernorm else nn.Identity()
        self.ln1 = nn.LayerNorm(hid) if self.use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(hid) if self.use_layernorm else nn.Identity()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hid + in_edge, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(hid, 1),
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        try:
            return super().load_state_dict(state_dict, *args, **kwargs)
        except RuntimeError:
            remapped = remap_legacy_state_dict(state_dict, self.state_dict())
            if remapped is None:
                raise
            return super().load_state_dict(remapped, *args, **kwargs)

    def forward(self, x_node: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln0(self.n_lin(x_node)))
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        h = F.relu(self.ln1(self.gat1(h, edge_index, edge_attr)))
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        h = F.relu(self.ln2(self.gat2(h, edge_index, edge_attr)))
        src, dst = edge_index
        e = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_mlp(e).squeeze(-1)
