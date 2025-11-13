import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Map textual status to numeric score
STATUS_MAP = {
    "trails": -1,
    "tied": 0,
    "leads": 1,
}


def parse_time_to_sec(s: str) -> float:
    """Convert 'MM:SS.s' to total seconds remaining in the quarter."""
    s = str(s)
    if ":" not in s:
        return 0.0
    m, rest = s.split(":")
    try:
        return int(m) * 60 + float(rest)
    except ValueError:
        return 0.0


def quarter_to_num(q: str) -> int:
    q = str(q).lower()
    if "1st" in q:
        return 1
    if "2nd" in q:
        return 2
    if "3rd" in q:
        return 3
    if "4th" in q:
        return 4
    if "ot" in q:
        # treat all overtimes as 5
        return 5
    return 0


def compute_score_diff(row) -> int:
    """
    score column is 'teamScore-oppScore' from the shooting team's perspective.
    We take teamScore - oppScore so negative means trailing.
    """
    s = str(row["score"])
    if "-" not in s:
        return 0
    a, b = s.split("-")
    try:
        a = int(a)
        b = int(b)
    except ValueError:
        return 0
    return a - b


def load_all_shot_files(data_dir: str, max_files: int | None = None) -> pd.DataFrame:
    """
    Load and concatenate all daily CSVs under data_dir.
    Files are named like 20001031.csv.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if max_files is not None:
        paths = paths[:max_files]

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        # Drop extraneous index columns
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)
    return big_df


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features to feed into the sequence dataset.

    Expected columns from the Kaggle dataset:
        match_id, shotX, shotY, quarter, time_remaining, player,
        team, made, shot_type, distance, score, opp, status
    """
    df = raw_df.copy()

    # Handle possible naming 'result' vs 'status'
    if "status" not in df.columns and "result" in df.columns:
        df = df.rename(columns={"result": "status"})

    df["shot_type"] = df["shot_type"].astype(str)

    # Keep only regular field goals
    keep_mask = df["shot_type"].isin(["2-pointer", "3-pointer"])
    df = df[keep_mask].copy()

    # ---------- Normalize 3-point distances ----------
    # NBA 3pt line is 23.75 ft; some threes are stored as 23.
    # For consistency, treat all 3-pointers as at least 24 ft.
    df.loc[df["shot_type"] == "3-pointer", "distance"] = 24.0

    # ---------- Time / quarter features ----------
    df["time_sec"] = df["time_remaining"].astype(str).apply(parse_time_to_sec)
    df["quarter_num"] = df["quarter"].astype(str).apply(quarter_to_num)

    # ---------- Game context ----------
    df["score_diff"] = df.apply(compute_score_diff, axis=1)
    df["made_int"] = df["made"].astype(int)
    df["status_int"] = df["status"].astype(str).map(STATUS_MAP).fillna(0)

    # Clean opponent string (strip quotes)
    df["opp"] = df["opp"].astype(str).str.replace("'", "", regex=False)

    # Player IDs as integers
    df["player_id"] = df["player"].astype("category").cat.codes

    # ---------- Distance-range label ----------
    # 0–5 ft      -> close range
    # 6–11 ft     -> short mid-range
    # 12–23 ft    -> long mid-range
    # 24+ ft      -> three-point range
    def distance_bucket(d):
        if pd.isna(d):
            return np.nan
        if d <= 5.0:
            return 0  # close
        elif d <= 11.0:
            return 1  # short mid-range
        elif d <= 23.0:
            return 2  # long mid-range
        else:
            return 3  # three-point range

    df["range_id"] = df["distance"].apply(distance_bucket)
    df = df.dropna(subset=["range_id"]).copy()
    df["range_id"] = df["range_id"].astype(int)

    return df


class ShotSequenceDataset(Dataset):
    """
    Builds sequences of length seq_len of a single player's shots.
    Label = distance-range class of the *next* shot:
        0: 0–5 ft          (close)
        1: 6–11 ft         (short mid-range)
        2: 12–23 ft        (long mid-range)
        3: 24+ ft          (three-point range)
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 10):
        self.seq_len = seq_len

        # Sort shots chronologically for each player
        df = df.sort_values(
            ["player_id", "match_id", "quarter_num", "time_sec"],
            ascending=[True, True, True, False],
        )

        self.feature_cols = [
            "shotX",
            "shotY",
            "distance",
            "time_sec",
            "quarter_num",
            "score_diff",
            "made_int",
            "status_int",
        ]

        # Normalize numeric features
        self.means = df[self.feature_cols].mean()
        self.stds = df[self.feature_cols].std() + 1e-6
        df[self.feature_cols] = (df[self.feature_cols] - self.means) / self.stds

        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []
        self.player_ids: list[int] = []

        for pid in df["player_id"].unique():
            block = df[df["player_id"] == pid]

            feats = block[self.feature_cols].values
            labels = block["range_id"].values
            pids = block["player_id"].values

            if len(block) <= seq_len:
                continue

            for i in range(len(block) - seq_len):
                self.sequences.append(feats[i : i + seq_len])
                self.labels.append(labels[i + seq_len])
                self.player_ids.append(pids[i + seq_len])

        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        self.player_ids = torch.tensor(np.array(self.player_ids), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.player_ids[idx]
