import argparse
import yaml
import pickle as pkl
import numpy as np
import torch

# funcoines de uso
from .transforms import ajustar_longitud_secuencia, centrar_escalar_esqueleto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument(
        '--input',
        required=True,
        help='ruta al .pkl de un video (formato UCF101 skeleton)'
    )
    ap.add_argument('--topk', type=int, default=3)
    args = ap.parse_args()

    # Cargar config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar checkpoint entrenado
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt['classes']

  #load pkl
    with open(args.input, 'rb') as f:
        obj = pkl.load(f)

    # Acepta dict o array directo
    if isinstance(obj, dict):
        datos = obj.get('data', obj.get('keypoint'))
        if datos is None:
            datos = None
            for v in obj.values():
                if isinstance(v, np.ndarray):
                    datos = v
                    break
            if datos is None:
                raise ValueError("No se encontro un ndarray en el pkl de entrada.")
    else:
        datos = obj

    arr = np.asarray(datos, dtype=np.float32)
    print("Shape original del input:", arr.shape)

    if arr.ndim == 4 and arr.shape[-1] == 2:
        # nos quedamos con la primera persona
        arr = arr[0]          
        print("Usando solo la persona 0, nuevo shape:", arr.shape)

    #Ahora esperamos algo 3D con última dim = 2
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"Shape inesperado después de procesar: {arr.shape}")

    # Puede venir [T,J,2] o [J,T,2]
    if arr.shape[1] == cfg['J'] and arr.shape[2] == 2:
        # ya es [T, J, 2]
        pass
    elif arr.shape[0] == cfg['J'] and arr.shape[2] == 2:
        # [J, T, 2] a [T, J, 2]
        arr = np.transpose(arr, (1, 0, 2))
    else:
        raise ValueError(f"Shape inesperado para J={cfg['J']}: {arr.shape}")

  

    arr = ajustar_longitud_secuencia(arr, cfg['T'])
    arr = centrar_escalar_esqueleto(
        arr,
        pelvis_index=cfg['pelvis_index'],
        height_pair=cfg['height_pair']
    )

    #[T, J, 2] -> [1, T, J*2]
    x = arr.reshape(arr.shape[0], -1)[None, ...]

    in_dim = cfg['J'] * 2
    num_classes = len(classes)


    #Reconstruir modelo
    from .models.improved_lstm import ImprovedLSTM
    from .models.baseline_lstm import BaselineLSTM

    model = ImprovedLSTM(in_dim, cfg['hidden_size'], cfg['num_layers'], num_classes)
    try:
        model.load_state_dict(ckpt['model_state'])
        modelo_usado = "ImprovedLSTM"
    except Exception:
        model = BaselineLSTM(in_dim, cfg['hidden_size'], cfg['num_layers'], num_classes)
        model.load_state_dict(ckpt['model_state'])
        modelo_usado = "BaselineLSTM"

    print("Usando modelo:", modelo_usado)

    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device).float())
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    
    #Mostrar top-k predicciones
  
    topk = min(args.topk, num_classes)
    idx = np.argsort(-probs)[:topk]
    print("\nTop-{} predictions:".format(topk))
    for i in idx:
        print(f"  clase {classes[i]}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()
