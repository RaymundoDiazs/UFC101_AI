import argparse
import os
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

from .datasets import SkeletonVideoDataset, make_splits


def _to_str_labels(labels):
    # Asegura que las etiquetas sean strings
    return [str(c) for c in labels]


def plot_confusion(cm: np.ndarray, classes, out_png: str, title: str = ""):
    classes = _to_str_labels(classes)
    n = cm.shape[0]

    plt.figure(figsize=(max(6, min(12, n * 0.5)), max(5, min(10, n * 0.4))))
    if _HAS_SNS and n <= 25:
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=False,
            square=False,
        )
    else:
        #else
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.xticks(ticks=np.arange(n), labels=classes, rotation=90)
        plt.yticks(ticks=np.arange(n), labels=classes)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]  # mismas clases usadas en train

    #Reproducir el split de validacion
    train_items, val_items = make_splits(
        cfg["data_root"], classes, cfg["val_ratio"], cfg["seed"]
    )

    val_ds = SkeletonVideoDataset(
        cfg["data_root"],
        classes,
        val_items,
        cfg["T"],
        cfg["J"],
        cfg["pelvis_index"],
        cfg["height_pair"],
        normalize=True,
    )
    num_workers = int(cfg.get("num_workers", 0))
    val_ld = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=num_workers
    )

    # Reconstruir modelo 
    in_dim = cfg["J"] * 2
    num_classes = len(classes)
    from .models.improved_lstm import ImprovedLSTM
    from .models.baseline_lstm import BaselineLSTM

    try:
        model = ImprovedLSTM(in_dim, cfg["hidden_size"], cfg["num_layers"], num_classes)
        model.load_state_dict(ckpt["model_state"])
    except Exception:
        model = BaselineLSTM(in_dim, cfg["hidden_size"], cfg["num_layers"], num_classes)
        model.load_state_dict(ckpt["model_state"])

    model = model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_ld:
            x = x.to(device).float()
            logits = model(x)
            y_hat = torch.argmax(logits, dim=-1).cpu().numpy()
            y_pred.append(y_hat)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"Val Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

    out_dir = os.path.dirname(args.ckpt)
    out_png = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion(cm, classes, out_png, title=f"Confusion Matrix (F1={f1:.3f}, Acc={acc:.3f})")
    print(f"Saved confusion matrix to: {out_png}")


if __name__ == "__main__":
    main()
