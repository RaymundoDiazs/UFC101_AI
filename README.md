Markdown

# Human Action Recognition (UCF101 Skeletons)

Este proyecto implementa un modelo de Deep Learning eficaz capaz de reconocer acciones humanas en videos del dataset UCF101 utilizando secuencias de esqueletos 2D.

---

## Instalación y Requisitos

**1. Clonar el repositorio:**
```bash
git clone [https://github.com/RaymundoDiazs/UFC101_AI.git](https://github.com/RaymundoDiazs/UFC101_AI.git)
cd UFC101_AI
2. Instalar dependencias:
```

Bash

```bash
pip install -r requirements.txt
```
Datos y Preparación
Dataset UCF101 Skeletons
Se utilizaron las anotaciones de esqueletos 2D contenidas en el archivo ucf101_2d.pkl.

NOTA IMPORTANTE: Debido a que el archivo original pesa más de 1GB, GitHub no permite alojarlo directamente. Se requiere un paso manual de preparación.

Paso 1: Preparar los Datos
Para optimizar la carga en PyTorch, se diseñó un script que divide el archivo masivo en archivos .pkl individuales.

Descarga el archivo ucf101_2d.pkl (disponible en Canvas) y colócalo en la raíz de este proyecto.

Ejecuta el script de preparación:

Bash

```bash
python prepare_data.py
```

Nota: Este script procesa un archivo y los divide de archivos individuales .pkl en la carpeta data/ucf101_skeletons/, necesarios para el entrenamiento e inferencia.

Modelos Implementados
El proyecto compara dos arquitecturas ubicadas en src/models/:

BaselineLSTM: Modelo estándar de redes recurrentes (LSTM + Capa Lineal).

ImprovedLSTM: Modelo optimizado para reducir el overfitting. Incluye:

Batch Normalization

Dropout (Regularización)

Ejecución
1. Inferencia (Predicción)
Para probar el modelo con un video específico usando el checkpoint pre-entrenado incluido (runs/baseline/best.pt):

Bash

```bash
python -m src.infer \
  --config configs/default.yaml \
  --ckpt runs/baseline/best.pt \
  --input data/ucf101_skeletons/0/0_v_ApplyEyeMakeup_g01_c01.pkl
(cambia la ruta --input por un archivo real generado en tu carpeta data/).
```

2. Entrenamiento
Para entrenar los modelos desde cero:

Bash

```bash
# Entrenar Baseline
python -m src.train --config configs/default.yaml --model baseline
```

# Entrenar Improved (con Dropout)
python -m src.train --config configs/default.yaml --model improved --dropout 0.4

3. Evaluación
Para obtener métricas del set de validación:

Bash
```bash
python -m src.eval --config configs/default.yaml --ckpt runs/baseline/best.pt
```