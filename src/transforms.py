import numpy as np


def ajustar_longitud_secuencia(arr: np.ndarray, T: int) -> np.ndarray:

    T0 = arr.shape[0]

    # ya está a la longitud deseada
    if T0 == T:
        return arr

    # recorte
    if T0 > T:
        return arr[:T]

    # repetir la secuencia
    repeticiones = (T + T0 - 1) // T0
    arr_repeat = np.concatenate([arr] * repeticiones, axis=0)

    return arr_repeat[:T]


def centrar_escalar_esqueleto(arr: np.ndarray, pelvis_index: int, height_pair):

    out = arr.copy()

    # Centrar con pelvis
    pelvis = out[:, pelvis_index:pelvis_index + 1, :]  # [T,1,2]
    out = out - pelvis

    # Calcular la distancia entre dos articulaciones clave
    j1, j2 = int(height_pair[0]), int(height_pair[1])
    distancias = np.linalg.norm(out[:, j1, :] - out[:, j2, :], axis=-1)

    # La mediana evita valores atípicos
    escala = np.median(distancias)
    if escala <= 1e-6:
        escala = 1.0

    # Normalizar toda la secuencia
    out = out / escala

    return out



# funciones antiguas fixeed
pad_or_crop_sequence = ajustar_longitud_secuencia
center_scale_skeleton = centrar_escalar_esqueleto
