import os
import glob
import pickle as pkl
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import centrar_escalar_esqueleto, ajustar_longitud_secuencia


class SkeletonVideoDataset(Dataset):
    """
    Dataset sencillo para videos representados como secuencias de esqueletos 2D.

    Cada archivo .pkl corresponde (idealmente) a UN video y contiene:
      - un array con forma [T, J, 2]  (T = frames, J = joints, 2 = (x, y))
      - o bien un dict con alguna de las llaves: 'data', 'keypoint', 'keypoints'.

    El __getitem__ devuelve:
      - features: tensor [T, J * 2]   (coordenadas aplanadas por joint)
      - label:    entero con el id de clase
    """

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
        """
        Parameters
        ----------
        data_root : str
            Carpeta raíz donde viven las subcarpetas por clase con .pkl.
        class_names : list[str]
            Nombres (o ids) de las clases que vamos a usar.
        split_items : list[ (path, label) ]
            Lista pre-generada con (ruta_al_pkl, id_clase).
        T : int
            Longitud temporal objetivo. Se hace pad/crop a este valor.
        J : int
            Número de articulaciones esperadas por frame.
        pelvis_index : int
            Índice de la pelvis (para centrar el esqueleto).
        height_pair : (int, int)
            Par de joints usado como “altura” para escalar (normalización).
        normalize : bool
            Si True, centra en pelvis y escala por altura.
        """
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
        """
        Carga un .pkl y lo regresa como array [T, J, 2].

        Soporta varias formas de guardar los keypoints:
          - dict con 'data', 'keypoint' o 'keypoints'
          - array con distintas formas: [T,J,2], [J,T,2], [N,T,J,2], [N,J,T,2]
        """
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
                # plan B: busca el primer ndarray dentro del dict
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

        # --- Normalización de formas ---
        # Queremos llegar siempre a algo tipo [T, J, 2]

        if arr.ndim == 3:
            # Puede ser [T, J, 2] o [J, T, 2]
            if arr.shape[1] == self.J and arr.shape[2] == 2:
                # Caso ideal: [T, J, 2]
                pass
            elif arr.shape[0] == self.J and arr.shape[2] == 2:
                # Caso [J, T, 2] -> transponemos a [T, J, 2]
                arr = np.transpose(arr, (1, 0, 2))
            else:
                raise ValueError(f"Shape inesperado {arr.shape} en {pkl_path}")

        elif arr.ndim == 4 and arr.shape[-1] == 2:
            # Casos con batch o múltiples personas: [N, T, J, 2] o [N, J, T, 2]
            if arr.shape[2] == self.J:
                # [N, T, J, 2] -> aplanamos N: [N*T, J, 2]
                arr = arr.reshape(-1, self.J, 2)
            elif arr.shape[1] == self.J:
                # [N, J, T, 2] -> [N, T, J, 2] -> [N*T, J, 2]
                arr = np.transpose(arr, (0, 2, 1, 3)).reshape(-1, self.J, 2)
            else:
                # Fallback: intentar encontrar el eje que coincide con J
                ejes = list(arr.shape)
                try:
                    j_axis = ejes.index(self.J)
                    # Reordenamos dejando (tiempo, joints, 2) al final
                    order = [i for i in range(4) if i not in (j_axis, 3)] + [j_axis, 3]
                    arr = np.transpose(arr, order)
                    arr = arr.reshape(-1, self.J, 2)
                except ValueError:
                    raise ValueError(f"No se pudo adaptar {arr.shape} en {pkl_path}")
        else:
            raise ValueError(f"Shape inesperado {arr.shape} en {pkl_path}")

        # En este punto arr debería ser [T, J, 2] (con T posiblemente muy grande)
        return arr

    def __getitem__(self, idx: int):
        """
        Devuelve una muestra lista para el modelo:
          x: tensor [T, J*2]   (coordenadas (x,y) aplanadas por articulación)
          y: label entero (long)
        """
        pkl_path, label_int = self.items_split[idx]

        # 1) Cargar y normalizar forma [T, J, 2]
        skeleton_seq = self._load_pkl(pkl_path)

        # 2) Recortar o repetir hasta longitud T objetivo
        skeleton_seq = ajustar_longitud_secuencia(skeleton_seq, self.T)

        # 3) Normalización opcional: centrar en pelvis y escalar por altura
        if self.normalize:
            skeleton_seq = centrar_escalar_esqueleto(
                skeleton_seq,
                pelvis_index=self.pelvis_index,
                height_pair=self.height_pair,
            )

        # 4) Aplanar por joint -> [T, J*2]
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
    """
    Genera listas de (filepath, label) para train y val de forma estratificada.

    Parameters
    ----------
    data_root : str
        Carpeta raíz donde viven las subcarpetas por clase.
    class_names : list[str]
        Lista de nombres/ids de clase. Deben coincidir con los nombres de carpeta.
    val_ratio : float
        Proporción de ejemplos que irán a validación por clase.
    seed : int
        Semilla para hacer el split reproducible.

    Returns
    -------
    train_items : list[(str, int)]
    val_items   : list[(str, int)]
        Listas con rutas a .pkl y el id entero de clase.
    """
    rng = np.random.RandomState(seed)
    label_por_nombre = {nombre: i for i, nombre in enumerate(class_names)}

    todos_los_videos: List[Tuple[str, int]] = []

    # Recorre cada carpeta de clase y junta los .pkl
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
