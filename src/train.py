"""
==============================================================
 Entrenamiento de modelos LSTM para reconocimiento de acciones
 mediante esqueletos humanos 2D (UCF101 Skeleton Dataset)
--------------------------------------------------------------
Este módulo:
 - Carga configuración desde un archivo YAML.
 - Prepara el dataset (train/validation).
 - Construye el modelo LSTM (baseline o improved).
 - Entrena por épocas, calcula métricas y guarda el mejor modelo.
==============================================================
"""

import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from .datasets import SkeletonVideoDataset, make_splits
from .models.baseline_lstm import BaselineLSTM
from .models.improved_lstm import ImprovedLSTM


# ============================================================
#  Utilidad para reproducibilidad
# ============================================================

def fijar_semilla(seed: int):
    """Fija semillas globales para reproducibilidad."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
#  Cross-Entropy con label smoothing
# ============================================================

def cross_entropy_label_smoothing(epsilon: float):
    """
    Cross Entropy con suavizado de etiquetas (label smoothing).
    Reduce overfitting y estabiliza el entrenamiento.
    """

    class _CELS(nn.Module):
        def __init__(self, eps: float):
            super().__init__()
            self.eps = eps
            self.log_softmax = nn.LogSoftmax(dim=-1)

        def forward(self, logits, targets):
            n_classes = logits.size(-1)
            log_probs = self.log_softmax(logits)

            with torch.no_grad():
                distrib_suave = torch.zeros_like(log_probs)
                distrib_suave.fill_(self.eps / (n_classes - 1))
                distrib_suave.scatter_(1, targets.unsqueeze(1), 1 - self.eps)

            return torch.mean(torch.sum(-distrib_suave * log_probs, dim=-1))

    return _CELS(epsilon)


# ============================================================
#  Entrenamiento por una época
# ============================================================

def entrenar_epoch(modelo, dataloader, optimizador, criterio, device):
    """
    Ejecuta UNA época de entrenamiento y devuelve:
    - pérdida promedio
    - accuracy
    - F1 macro
    """
    modelo.train()
    perdidas, preds, etiquetas = [], [], []

    for batch_x, batch_y in tqdm(dataloader, desc="Entrenando", leave=False):
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device)

        optimizador.zero_grad()
        logits = modelo(batch_x)
        loss = criterio(logits, batch_y)
        loss.backward()
        optimizador.step()

        perdidas.append(loss.item())
        preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
        etiquetas.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds)
    etiquetas = np.concatenate(etiquetas)

    return float(np.mean(perdidas)), accuracy_score(etiquetas, preds), f1_score(etiquetas, preds, average="macro")


# ============================================================
#  Evaluación en validación
# ============================================================

def evaluar(modelo, dataloader, criterio, device):
    """Evalúa el modelo en el conjunto de validación."""
    modelo.eval()
    perdidas, preds, etiquetas = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Validando", leave=False):
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device)

            logits = modelo(batch_x)
            loss = criterio(logits, batch_y)

            perdidas.append(loss.item())
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
            etiquetas.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds)
    etiquetas = np.concatenate(etiquetas)

    return float(np.mean(perdidas)), accuracy_score(etiquetas, preds), f1_score(etiquetas, preds, average="macro")


# ============================================================
#  Entrenamiento principal
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta al archivo YAML")
    ap.add_argument("--model", default="baseline", choices=["baseline", "improved"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--label_smoothing", type=float)
    ap.add_argument("--dropout", type=float)
    args = ap.parse_args()

    # ------------------------------
    # Cargar configuración
    # ------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    fijar_semilla(cfg.get("seed", 42))

    # ------------------------------
    # Preparar dataset
    # ------------------------------
    data_root = cfg["data_root"]
    clases = cfg["classes"]

    items_train, items_val = make_splits(data_root, clases, cfg["val_ratio"], cfg["seed"])

    T = cfg["T"]
    J = cfg["J"]
    pelvis = cfg["pelvis_index"]
    par_altura = cfg["height_pair"]

    datos_entrenamiento = SkeletonVideoDataset(data_root, clases, items_train, T, J, pelvis, par_altura, normalize=True)
    datos_validacion   = SkeletonVideoDataset(data_root, clases, items_val,   T, J, pelvis, par_altura, normalize=True)

    num_workers = cfg.get("num_workers", 0)
    pin_memory = cfg.get("pin_memory", False)
    batch_size = args.batch_size or cfg["batch_size"]

    dl_train = DataLoader(datos_entrenamiento, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    dl_val   = DataLoader(datos_validacion, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    # ------------------------------
    # Construcción del modelo
    # ------------------------------
    in_dim = J * 2
    hidden = cfg["hidden_size"]
    capas = cfg["num_layers"]
    num_clases = len(clases)

    dropout = args.dropout if args.dropout is not None else cfg.get("dropout", 0.0)

    if args.model == "baseline":
        modelo = BaselineLSTM(in_dim, hidden, capas, num_clases)
        run_dir = "runs/baseline"
    else:
        modelo = ImprovedLSTM(in_dim, hidden, capas, num_clases, dropout=dropout)
        run_dir = "runs/improved"

    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)

    # ------------------------------
    # Optimizador y loss
    # ------------------------------
    lr = args.lr or cfg["lr"]
    epochs = args.epochs or cfg["epochs"]

    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-4)

    smoothing_val = args.label_smoothing if args.label_smoothing is not None else cfg.get("label_smoothing", 0.0)
    criterio = cross_entropy_label_smoothing(smoothing_val) if smoothing_val > 0 else nn.CrossEntropyLoss()

    # ------------------------------
    # Loop de entrenamiento
    # ------------------------------
    mejor_f1 = -1.0
    ruta_best = os.path.join(run_dir, "best.pt")

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = entrenar_epoch(modelo, dl_train, optimizador, criterio, device)
        vl_loss, vl_acc, vl_f1 = evaluar(modelo, dl_val, criterio, device)

        print(f"[Ep {ep:03d}] "
              f"train → loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
              f"val → loss={vl_loss:.4f} acc={vl_acc:.3f} f1={vl_f1:.3f}")

        if vl_f1 > mejor_f1:
            mejor_f1 = vl_f1
            torch.save(
                {"model_state": modelo.state_dict(), "classes": clases, "cfg": cfg},
                ruta_best
            )
            print(f"   💾 Nuevo mejor modelo (F1={mejor_f1:.3f}) → {ruta_best}")


if __name__ == "__main__":
    main()
