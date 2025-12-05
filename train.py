import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data_preprocessing import load_all_shot_files, prepare_dataframe, ShotSequenceDataset
from training_models import RNNClassifier, TCNClassifier, TransformerClassifier
from losses import compute_player_baselines, BaselineWeightedCE


def create_splits(df, test_size=0.15, val_size=0.15, seed=42):
    """
    Split by players so val/test players are unseen at training time.
    """
    players = df["player_id"].unique()
    train_players, temp_players = train_test_split(
        players, test_size=test_size + val_size, random_state=seed
    )

    # split temp into val / test
    val_rel = val_size / (test_size + val_size)
    val_players, test_players = train_test_split(
        temp_players, test_size=1 - val_rel, random_state=seed
    )

    train_df = df[df["player_id"].isin(train_players)]
    val_df = df[df["player_id"].isin(val_players)]
    test_df = df[df["player_id"].isin(test_players)]
    return train_df, val_df, test_df


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y, pids in loader:
        x, y, pids = x.to(device), y.to(device), pids.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y, pids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, _pids in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1


def run_experiment(
    model_name="transformer",
    data_dir="data/train",
    seq_len=10,
    epochs=5,
    batch_size=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load & prepare data ----------
    raw_df = load_all_shot_files(data_dir)
    df = prepare_dataframe(raw_df)

    train_df, val_df, test_df = create_splits(df)

    # Player baselines computed on train set
    player_baselines = compute_player_baselines(train_df)

    train_ds = ShotSequenceDataset(train_df, seq_len=seq_len)
    val_ds = ShotSequenceDataset(val_df, seq_len=seq_len)
    test_ds = ShotSequenceDataset(test_df, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = len(train_ds.feature_cols)

    # ---------- Choose model ----------
    if model_name == "rnn":
        model = RNNClassifier(input_dim=input_dim)
    elif model_name == "tcn":
        model = TCNClassifier(input_dim=input_dim)
    else:
        model = TransformerClassifier(input_dim=input_dim)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BaselineWeightedCE(player_baselines)

    # ---------- Training loop ----------
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print("Debug: This line works")
        print(
            f"[{model_name}] Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_acc={val_acc:.3f}, val_macroF1={val_f1:.3f}"
        )

    # ---------- Final test evaluation ----------
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"[{model_name}] FINAL TEST: acc={test_acc:.3f}, macroF1={test_f1:.3f}")

    return model


if __name__ == "__main__":
    # Example: run all three models for comparison
    for name in ["rnn", "tcn", "transformer"]:
        print(f"\n=== Running {name.upper()} ===")
        run_experiment(model_name=name, data_dir="data/nba_shots", seq_len=10, epochs=5)
