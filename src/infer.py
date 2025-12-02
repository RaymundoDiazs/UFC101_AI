import argparse
import yaml
import pickle as pkl
import numpy as np
import torch

from .transforms import pad_or_crop_sequence, center_scale_skeleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input', required=True, help='ruta al .pkl de un video')
    ap.add_argument('--topk', type=int, default=3)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt['classes']

    #Cargar sample
    with open(args.input, 'rb') as f:
        obj = pkl.load(f)
    if isinstance(obj, dict):
        data = obj.get('data', obj.get('keypoints'))
        if data is None:
            # usa primer ndarray
            data = None
            for v in obj.values():
                if isinstance(v, np.ndarray):
                    data = v
                    break
            if data is None:
                raise ValueError("No se encontró un ndarray en el pkl de entrada.")
    else:
        data = obj

    arr = np.asarray(data, dtype=np.float32)
    # reacomodo si viene [J,T,2]
    if arr.ndim == 3 and arr.shape[2] == 2 and arr.shape[1] == cfg['J']:
        pass  # ya es [T,J,2]
    elif arr.ndim == 3 and arr.shape[0] == cfg['J'] and arr.shape[2] == 2:
        arr = np.transpose(arr, (1, 0, 2))  # [J,T,2] a [T,J,2]
    else:
        raise ValueError(f"Shape inesperado: {arr.shape}")

    arr = pad_or_crop_sequence(arr, cfg['T'])
    arr = center_scale_skeleton(arr, pelvis_index=cfg['pelvis_index'], height_pair=cfg['height_pair'])
    x = arr.reshape(arr.shape[0], -1)[None, ...]  # [1,T,J*2]

    in_dim = cfg['J'] * 2
    num_classes = len(classes)

    # reconstruir modelo
    from .models.improved_lstm import ImprovedLSTM
    from .models.baseline_lstm import BaselineLSTM

    model = ImprovedLSTM(in_dim, cfg['hidden_size'], cfg['num_layers'], num_classes)
    try:
        model.load_state_dict(ckpt['model_state'])
    except Exception:
        model = BaselineLSTM(in_dim, cfg['hidden_size'], cfg['num_layers'], num_classes)
        model.load_state_dict(ckpt['model_state'])

    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device).float())
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    topk = min(args.topk, num_classes)
    idx = np.argsort(-probs)[:topk]
    print("Top-{} predictions:".format(topk))
    for i in idx:
        print(f"  {classes[i]}: {probs[i]*100:.2f}%")


if __name__ == '__main__':
    main()
