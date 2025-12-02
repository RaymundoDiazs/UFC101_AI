import os
import glob
import pickle as pkl
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import centrar_escalar_esqueleto, ajustar_longitud_secuencia


class SkeletonVideoDataset(Dataset):

    #cada .pkl con un array con forma [T, J, 2]  (T = frames, J = joints, 2 = (x, y))

    def __init__(
        self,
        data_root: str,
        class_names: List[str],
        split_items: List[Tuple[str, int]],
        T: int,
        J: int,
        pelvis_index: int,
        height_pair: Tuple[int, int],
        normalize: bool = True,
    ):

        self.data_root = data_root
        self.class_names = class_names
        self.label_map = {nombre: i for i, nombre in enumerate(class_names)}

        # Cada elemento de items_split es: (filepath, label_int)
        self.items_split = split_items

        self.T = T
        self.J = J
        self.pelvis_index = pelvis_index
        self.height_pair = height_pair
        self.normalize = normalize

    # Para DataLoader: len(dataset)
    def __len__(self) -> int:
        return len(self.items_split)

    def _load_pkl(self, pkl_path: str) -> np.ndarray:
        #carga el .pkl y lo regresa como array

        #fuarda los keypoins 

        with open(pkl_path, "rb") as f:
            objeto = pkl.load(f)

        # Acepta dict o array directo
        if isinstance(objeto, dict):
            if "data" in objeto:
                arr = objeto["data"]
            elif "keypoint" in objeto:
                arr = objeto["keypoint"]
            elif "keypoints" in objeto:
                arr = objeto["keypoints"]
            else:
                #
                arr = None
                for valor in objeto.values():
                    if isinstance(valor, np.ndarray):
                        arr = valor
                        break
                if arr is None:
                    raise ValueError(f"Formato pkl desconocido: {pkl_path}")
        else:
            # el pickle era directamente un array o lista
            arr = objeto

        arr = np.asarray(arr, dtype=np.float32)


        # normalizamos = buscamos [T, J, 2]

        if arr.ndim == 3:
            # puede ser [T, J, 2] o [J, T, 2]
            if arr.shape[1] == self.J and arr.shape[2] == 2:
                pass
            elif arr.shape[0] == self.J and arr.shape[2] == 2:
                arr = np.transpose(arr, (1, 0, 2))
            else:
                raise ValueError(f"Shape inesperado {arr.shape} en {pkl_path}")

        elif arr.ndim == 4 and arr.shape[-1] == 2:
            # multiples personas: [N, T, J, 2] o [N, J, T, 2]
            if arr.shape[2] == self.J:
                #[N, T, J, 2] -a a N: [N*T, J, 2]
                arr = arr.reshape(-1, self.J, 2)
            elif arr.shape[1] == self.J:
                arr = np.transpose(arr, (0, 2, 1, 3)).reshape(-1, self.J, 2)
            else:
                ejes = list(arr.shape)
                try:
                    j_axis = ejes.index(self.J)
                    order = [i for i in range(4) if i not in (j_axis, 3)] + [j_axis, 3]
                    arr = np.transpose(arr, order)
                    arr = arr.reshape(-1, self.J, 2)
                except ValueError:
                    raise ValueError(f"No se pudo adaptar {arr.shape} en {pkl_path}")
        else:
            raise ValueError(f"Shape inesperado {arr.shape} en {pkl_path}")

        return arr

    def __getitem__(self, idx: int):

        #devuelve una muestra de lista para el modelo 
        #x tesnsor
        #y label
    
        pkl_path, label_int = self.items_split[idx]

        #Cargamos y normalizar forma [T, J, 2]
        skeleton_seq = self._load_pkl(pkl_path)

        #Recortar o repetir hasta longitud T objetivo
        skeleton_seq = ajustar_longitud_secuencia(skeleton_seq, self.T)

        # centrar en pelvis y escalar por altura
        if self.normalize:
            skeleton_seq = centrar_escalar_esqueleto(
                skeleton_seq,
                pelvis_index=self.pelvis_index,
                height_pair=self.height_pair,
            )

        num_frames = skeleton_seq.shape[0]
        features = skeleton_seq.reshape(num_frames, -1)

        x_tensor = torch.from_numpy(features)              # [T, J*2]
        y_tensor = torch.tensor(label_int, dtype=torch.long)

        return x_tensor, y_tensor


def make_splits(
    data_root: str,
    class_names: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
):
   
    rng = np.random.RandomState(seed)
    label_por_nombre = {nombre: i for i, nombre in enumerate(class_names)}

    todos_los_videos: List[Tuple[str, int]] = []

    #Recorre cada carpeta de clase y junta los .pkl
    for nombre_clase in class_names:
        pattern = os.path.join(data_root, nombre_clase, "*.pkl")
        archivos = sorted(glob.glob(pattern))
        for ruta in archivos:
            todos_los_videos.append((ruta, label_por_nombre[nombre_clase]))

    # Agrupar por id de clase para hacer split estratificado
    from collections import defaultdict

    videos_por_clase = defaultdict(list)
    for ruta, label_int in todos_los_videos:
        videos_por_clase[label_int].append(ruta)

    train_items, val_items = [], []

    for label_int, lista_rutas in videos_por_clase.items():
        idx = np.arange(len(lista_rutas))
        rng.shuffle(idx)

        n_val = max(1, int(len(lista_rutas) * val_ratio))
        indices_val = set(idx[:n_val])

        for i, ruta in enumerate(lista_rutas):
            destino = val_items if i in indices_val else train_items
            destino.append((ruta, label_int))

    return train_items, val_items
