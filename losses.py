import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def compute_player_baselines(df: pd.DataFrame) -> dict[int, float]:
    """
    For each player_id, compute the accuracy of a naive baseline:
      always predict that player's most frequent distance-range class.
    Returns dict: player_id -> baseline_accuracy.
    """
    baselines: dict[int, float] = {}
    for pid, block in df.groupby("player_id"):
        counts = block["range_id"].value_counts()
        top_count = counts.max()
        total = counts.sum()
        baselines[int(pid)] = float(top_count) / float(total)
    return baselines


class BaselineWeightedCE(nn.Module):
    """
    Cross-entropy loss where each example is weighted by 1 / baseline(player).

    This discourages the model from coasting on trivially predictable players
    and emphasizes players with more diverse shot distance behavior.
    """

    def __init__(self, player_baselines: dict[int, float], epsilon: float = 1e-3):
        super().__init__()
        max_pid = max(player_baselines.keys())
        weights = np.ones(max_pid + 1, dtype=np.float32)

        for pid, b in player_baselines.items():
            weights[int(pid)] = 1.0 / (b + epsilon)

        self.player_weights = torch.tensor(weights)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, player_ids):
        """
        logits: (B, C)
        targets: (B,)
        player_ids: (B,) integer ids aligned with player_weights
        """
        losses = self.ce(logits, targets)  # (B,)
        w = self.player_weights[player_ids.cpu()].to(losses.device)
        return (losses * w).mean()
