

import pickle
import glob
import random
import os
import numpy as np
import matplotlib.pyplot as plt


#  Seleccionar un archivo .pkl aleatorio del dataset


rutas_pkl = glob.glob("data/ucf101_skeletons/*/*.pkl")
if not rutas_pkl:
    raise FileNotFoundError("No se encontraron archivos .pkl en data/ucf101_skeletons/")

ruta_sample = random.choice(rutas_pkl)
print(f"Mostrando ejemplo: {ruta_sample}")


# Cargar la secuencia y asegurar forma [T, J, 2]


obj = pickle.load(open(ruta_sample, "rb"))

#lee del keypoint
keypoints = obj.get("data", obj.get("keypoint"))
if keypoints is None:
    raise ValueError(f"El archivo no tiene keypoints válidos: {ruta_sample}")

arr = np.asarray(keypoints, dtype="float32")

#
if arr.ndim == 4:
    arr = arr[0]
elif arr.ndim != 3:
    raise ValueError(f"Forma inesperada al visualizar: {arr.shape}")

skel = arr  # [T, J, 2]
num_frames = skel.shape[0]



#elege tres frames 


frames_mostrar = np.unique(np.linspace(0, num_frames - 1, 3, dtype=int))

#dupes
if frames_mostrar.size < 3 and num_frames >= 3:
    frames_mostrar = np.array([0, num_frames // 2, num_frames - 1])

#conexiones

conexiones = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # brazos
    (11, 13), (13, 15), (12, 14), (14, 16), # piernas
    (5, 6), (11, 12), (5, 11), (6, 12)      # torso / hombros / cadera
]



# dibujar los esqueletos

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, frame_idx in zip(axes, frames_mostrar):
    #puntos del esqueleto en el frame
    puntos = skel[frame_idx]  # [J, 2]

    #dibujar articulaciones 
    ax.scatter(puntos[:, 0], -puntos[:, 1], s=20, color="red")

    #dibujar conexiones 
    for j1, j2 in conexiones:
        ax.plot(
            [puntos[j1, 0], puntos[j2, 0]],
            [-puntos[j1, 1], -puntos[j2, 1]],
            color="blue",
            linewidth=2
        )

    ax.set_title(f"Frame {frame_idx}")
    ax.axis("equal")
    ax.axis("off")

fig.suptitle(
    f"Skeleton Visualization\n{os.path.basename(ruta_sample)}",
    fontsize=12
)

plt.tight_layout()


# 6) Guardar la imagen en la carpeta runs


output_dir = "runs/visualizations"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "skeleton_triptych.png")
plt.savefig(output_path, dpi=150)

print(f"Imagen guardada en: {output_path}")

plt.show()
