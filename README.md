# Gyroscope Orientation Detection with YOLOv8-Pose + Hailo-8 on Raspberry Pi 5

> **Detección de Orientación de Giroscopio con YOLOv8-Pose + Hailo-8 en Raspberry Pi 5**

![Demo](assets/demo.gif)

**Author / Autor:** José Luis Guerrero Marín  
**Center / Centro:** IES Politécnico Jesús Marín (Málaga, España)  
**Project / Proyecto:** Innovación Educativa — Elaboración de Materiales — Junta de Andalucía  
**Blog:** [Industria 4.0](https://blogsaverroes.juntadeandalucia.es/industria4/)  
**YouTube:** [Presentación del proyecto](https://youtu.be/m3AsXro8m2U)  
**License / Licencia:** [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

---

## 🇬🇧 English

### What is this?

A real-time gyroscope orientation estimation system (Roll, Pitch, Yaw) built entirely on a Raspberry Pi 5 with the Hailo-8 AI accelerator. No cloud, no GPU. The system detects 9 keypoints on a physical drone gyroscope using a custom-trained YOLOv8n-Pose model running at ~25-30 FPS.

This project was developed as educational material for advanced vocational training (FP) cycles in Electronics and Robotics, demonstrating the complete pipeline from data collection to edge AI inference.

### Demo

| Detection | Angles panel |
|-----------|-------------|
| Keypoints distributed across gyroscope structure | Roll, Pitch, Yaw displayed in real time |

### Hardware required

| Component | Model |
|-----------|-------|
| SBC | Raspberry Pi 5 (4GB or 8GB) |
| AI accelerator | Raspberry Pi AI Kit (Hailo-8) |
| Camera | Raspberry Pi Camera Module v2 (IMX219) |
| 3D printed gyroscope | See `3d/` folder |

### Pipeline

```
Video recording → Frame extraction → Labeling (Roboflow) → Training (Colab)
      ↓
   best.pt  →  best_hailo.onnx  →  giroscopio_pose_v2.hef  →  Inference on Pi
  (PyTorch)      (ONNX)            (Hailo compiled)              ~25-30 FPS
```

### Repository structure

```
├── 3d/                          # Gyroscope 3D model (.stl) — designed for this project
├── docs/                        # Tutorial, slides, video presentation
└── sw/
    ├── 00 entrenamiento con Google Colab/   # Training notebook
    ├── 01 modelo compilado para PyTorch/    # best.pt (PyTorch model)
    ├── 02 modelo compilado ONNX exportado/  # best_hailo.onnx
    ├── 03 modelo compilado para Raspberry/  # giroscopio_pose_v2.hef
    ├── 04 script de inferencia para Raspberry/ # Inference script + commands
    └── archive/                             # Early versions
```

**Dataset:** Available on Roboflow — https://app.roboflow.com/joss-workspace-7orjh/giroscopio-dron/1

**Additional 3D models used (not included — third party):**
- Camera mount for Raspberry Pi Camera: https://cults3d.com/es/pedidos/151998246
- VESA mount for Raspberry Pi 4: https://cults3d.com/es/modelo-3d/artilugios/vesa-raspberry-pi-mount

### How to replicate

**Step 1 — Train the model**  
Download the dataset from Roboflow: https://app.roboflow.com/joss-workspace-7orjh/giroscopio-dron/1 (168 original images, augmented to 404). Then open `sw/00 entrenamiento con Google Colab/giroscopio_colab_v2.ipynb` in Google Colab, train YOLOv8n-Pose and export to ONNX. Training stopped early at epoch 131/150 (EarlyStopping, best model at epoch 101), completing in 0.3 hours on Colab GPU.

**Results at best epoch (101):** Box mAP50=0.983 · Pose mAP50=0.939 · Pose mAP50-95=0.885

**Step 2 — Compile for Hailo-8**  
Use the Hailo DFC (Dataflow Compiler) inside Docker to compile the ONNX to HEF. The tutorial in `docs/` covers this step in detail, including the workaround for the calibration bug in DFC 3.33.0.

**Step 3 — Run inference on Raspberry Pi**  
```bash
# Install dependencies
sudo apt install hailo-all -y
pip install picamera2 opencv-python numpy --break-system-packages

# Copy the HEF and script to /home/ai/
# Run
DISPLAY=:0 python3 giroscopio_pose_v2.py

# Record output video
DISPLAY=:0 python3 giroscopio_pose_v2.py --save /home/ai/resultado.mp4
```

See `sw/04 script de inferencia para Raspberry/COMANDOS.md` for the full command reference.

### Technical challenges solved

- **Hailo DFC calibration bug** — The CLI `hailo optimize` ignores numpy calibration data in DFC 3.33.0. Solution: use the Python API inside Docker directly.
- **RGB/BGR channel order** — Picamera2 delivers RGB888. The model was trained with BGR (OpenCV default). Swapping channels was the key fix for correct detection.
- **Keypoint decoder** — YOLOv8-Pose keypoints use `sigmoid(x) * 10 - 5` offset relative to the grid cell center, scaled by stride. The standard `* 4 - 2` formula produced clustered keypoints.
- **Letterbox preprocessing** — 960×540 camera resolution scaled to 640×640 with padding to avoid distortion, with matching coordinate inverse transform for display.

---

## 🇪🇸 Español

### ¿Qué es esto?

Un sistema de estimación de orientación de giroscopio en tiempo real (Roll, Pitch, Yaw) que funciona íntegramente en una Raspberry Pi 5 con el acelerador de IA Hailo-8. Sin nube, sin GPU. El sistema detecta 9 keypoints sobre un giroscopio físico de dron usando un modelo YOLOv8n-Pose entrenado específicamente, a ~25-30 FPS.

El proyecto se desarrolló como material didáctico para ciclos superiores de FP de Electrónica y Robótica, demostrando el pipeline completo desde la captura de datos hasta la inferencia en hardware embebido.

### Hardware necesario

| Componente | Modelo |
|------------|--------|
| SBC | Raspberry Pi 5 (4GB u 8GB) |
| Acelerador IA | Raspberry Pi AI Kit (Hailo-8) |
| Cámara | Raspberry Pi Camera Module v2 (IMX219) |
| Giroscopio impreso en 3D | Ver carpeta `3d/` |

### Pipeline

```
Grabación de vídeo → Extracción de frames → Etiquetado (Roboflow) → Entrenamiento (Colab)
         ↓
      best.pt  →  best_hailo.onnx  →  giroscopio_pose_v2.hef  →  Inferencia en Pi
    (PyTorch)       (ONNX)             (compilado para Hailo)         ~25-30 FPS
```

### Estructura del repositorio

```
├── 3d/                          # Modelo 3D del giroscopio (.stl) — diseñado para este proyecto
├── docs/                        # Tutorial, presentación, vídeo
└── sw/
    ├── 00 entrenamiento con Google Colab/   # Notebook de entrenamiento
    ├── 01 modelo compilado para PyTorch/    # best.pt
    ├── 02 modelo compilado ONNX exportado/  # best_hailo.onnx
    ├── 03 modelo compilado para Raspberry/  # giroscopio_pose_v2.hef
    ├── 04 script de inferencia para Raspberry/ # Script + comandos
    └── archive/                             # Versiones iniciales
```

**Dataset:** Disponible en Roboflow — https://app.roboflow.com/joss-workspace-7orjh/giroscopio-dron/1

**Modelos 3D adicionales utilizados (no incluidos — son de terceros):**
- Soporte para cámara Raspberry Pi: https://cults3d.com/es/pedidos/151998246
- Soporte VESA para Raspberry Pi 4: https://cults3d.com/es/modelo-3d/artilugios/vesa-raspberry-pi-mount

### Cómo replicarlo

**Paso 1 — Entrenar el modelo**  
Descarga el dataset desde Roboflow: https://app.roboflow.com/joss-workspace-7orjh/giroscopio-dron/1 (168 imágenes originales, aumentadas hasta 404). Luego abre `sw/00 entrenamiento con Google Colab/giroscopio_colab_v2.ipynb` en Google Colab, entrena YOLOv8n-Pose y exporta a ONNX. El entrenamiento se detuvo por EarlyStopping en la época 131/150 (mejor modelo en época 101), completándose en 0.3 horas en la GPU de Colab.

**Resultados en la mejor época (101):** Box mAP50=0.983 · Pose mAP50=0.939 · Pose mAP50-95=0.885

**Paso 2 — Compilar para Hailo-8**  
Usa el compilador DFC de Hailo dentro de Docker para compilar el ONNX a HEF. El tutorial en `docs/` cubre este paso en detalle, incluyendo el workaround para el bug de calibración del DFC 3.33.0.

**Paso 3 — Ejecutar inferencia en Raspberry Pi**  
```bash
# Instalar dependencias
sudo apt install hailo-all -y
pip install picamera2 opencv-python numpy --break-system-packages

# Copiar el HEF y el script a /home/ai/
# Lanzar
DISPLAY=:0 python3 giroscopio_pose_v2.py

# Con grabación de vídeo
DISPLAY=:0 python3 giroscopio_pose_v2.py --save /home/ai/resultado.mp4
```

Consulta `sw/04 script de inferencia para Raspberry/COMANDOS.md` para la referencia completa de comandos.

### Problemas técnicos resueltos

- **Bug de calibración del DFC de Hailo** — El CLI `hailo optimize` ignora los datos de calibración numpy en la versión DFC 3.33.0. Solución: usar la API Python dentro del Docker directamente.
- **Orden de canales RGB/BGR** — Picamera2 entrega RGB888. El modelo fue entrenado con BGR (por defecto en OpenCV). Intercambiar los canales fue la corrección clave para que la detección funcionara correctamente.
- **Decoder de keypoints** — Los keypoints de YOLOv8-Pose usan un offset `sigmoid(x) * 10 - 5` relativo al centro de la celda de la cuadrícula, escalado por el stride. La fórmula estándar `* 4 - 2` producía keypoints agrupados en el centro.
- **Preprocesado letterbox** — La resolución de la cámara (960×540) se escala a 640×640 con relleno negro para no deformar la imagen, con la transformación inversa correspondiente para dibujar en pantalla.

### Contexto educativo

Este proyecto forma parte de un proyecto de Innovación Educativa en la modalidad de Elaboración de Materiales, convocatoria de la Junta de Andalucía. El objetivo es que el alumnado de ciclos superiores de FP comprenda y pueda replicar un pipeline completo de visión artificial sobre hardware embebido real, desde la captura de datos hasta la inferencia en producción.

El tutorial completo (65 páginas) está disponible en `docs/tutorial_giroscopio_v6.pdf`.

---

## Keypoints defined / Keypoints definidos

| # | Name / Nombre | Location / Ubicación |
|---|---------------|----------------------|
| 0 | centro_base | Central axis joint / Unión central del eje |
| 1 | ext_der_rojo | Right horizontal support / Soporte horizontal derecho |
| 2 | ext_izq_rojo | Left horizontal support / Soporte horizontal izquierdo |
| 3 | frente_octogono | Front octagon joint / Unión frontal del octógono |
| 4 | trasero_octogono | Rear octagon joint / Unión trasera del octógono |
| 5 | pivote_der | Right lower pivot / Pivote inferior derecho |
| 6 | pivote_izq | Left lower pivot / Pivote inferior izquierdo |
| 7 | esquina_der_azul | Front-right base corner / Esquina frontal-derecha de la base |
| 8 | esquina_izq_azul | Front-left base corner / Esquina frontal-izquierda de la base |

---

## Citation / Cita

Si usas este material en tu trabajo, por favor cítalo como:  
If you use this material in your work, please cite it as:

```
Guerrero Marín, J.L. (2026). Gyroscope Orientation Detection with YOLOv8-Pose and Hailo-8
on Raspberry Pi 5. IES Politécnico Jesús Marín, Málaga.
https://github.com/[tu-usuario]/gyroscope-orientation-yolov8
```

---

*Developed with / Desarrollado con: YOLOv8 · Hailo-8 · Raspberry Pi 5 · Roboflow · Google Colab · Claude (Anthropic)*
