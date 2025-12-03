import os
import pickle
import numpy as np
import sys

#rutas
SRC_FILE = "ucf101_2d.pkl"  # El archivo original descargado
DST_ROOT = "data/ucf101_skeletons"

def main():
    if not os.path.exists(SRC_FILE):
        print(f"Error: No se encuentra el archivo '{SRC_FILE}' ")
        print("Descargar dataset y colocarlo")
        sys.exit(1)

    print(f"Leyendo {SRC_FILE} ")
    with open(SRC_FILE, "rb") as f:
        obj = pickle.load(f)

    anotaciones = obj["annotations"]
    contador = 0
    
    print(f"Procesando {len(anotaciones)} videos...")
    
    # Crear carpeta destino
    os.makedirs(DST_ROOT, exist_ok=True)

    for i, item in enumerate(anotaciones):
        # float para menor espacio
        keypoints = np.asarray(item["keypoint"], dtype="float32")  # [M,T,V,2]
        label = str(item["label"])
        frame_dir = str(item.get("frame_dir", f"sample_{i:06d}"))

        #
        if keypoints.ndim == 4:
            keypoints = keypoints[0]  # [T,V,2]

        #carpeta de clase
        out_dir = os.path.join(DST_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(out_dir, f"{label}_{frame_dir}.pkl")

        #guardamos en formato diccionario simplificado
        with open(out_path, "wb") as f:
            pickle.dump({"data": keypoints}, f)

        contador += 1
        
        if i % 500 == 0:
            print(f"Procesados: {i}")

    print(f"\nSe generaron {contador} archivos .pkl en '{DST_ROOT}'.")

if __name__ == "__main__":
    main()