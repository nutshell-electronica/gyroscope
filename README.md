# Gyroscope Edge AI — YOLOv8n-Pose + Hailo-8 on Raspberry Pi 5

**Detección en tiempo real de la orientación de un giroscopio mecánico mediante 12 keypoints
personalizados, ejecutado íntegramente en una Raspberry Pi 5 con NPU Hailo-8 — sin nube, sin GPU.**

> Proyecto de innovación educativa MTT-041/25 · IES Politécnico Jesús Marín, Málaga  
> CFGS Mantenimiento Electrónico · Curso de Especialización en Mantenimiento de Drones  
> Autor: José Luis Guerrero Marín · Junio 2026

---

## ¿Qué es esto?

Un sistema de visión artificial que detecta un giroscopio mecánico de dron y estima sus
ángulos de orientación (Roll, Pitch, Yaw) en tiempo real. El modelo YOLOv8n-Pose detecta
12 keypoints personalizados distribuidos en tres grupos estructurales del giroscopio, y a
partir de sus coordenadas 2D calcula la orientación tridimensional.

Todo el pipeline —captura, inferencia, cálculo y visualización— corre en una
**Raspberry Pi 5 con AI HAT+ (chip Hailo-8, 26 TOPS)** sin conexión a internet ni GPU externa,
a ~22 FPS con el modelo de producción.

El proyecto tiene un doble objetivo: ser un **banco de trabajo didáctico real** para alumnos
de electrónica, y demostrar que el pipeline completo de Edge AI
(dataset → etiquetado → entrenamiento → compilación → inferencia embebida)
es **replicable por estudiantes de FP** con herramientas gratuitas.

> 🌐 [English version → README_EN.md](README_EN.md)

![Demo](assets/demo.gif)

---

## Demo

> 🎬 *Vídeo de presentación — próximamente*

📖 Blog del proyecto:
[blogsaverroes.juntadeandalucia.es/industria4](https://blogsaverroes.juntadeandalucia.es/industria4/2026/03/15/deteccion-de-orientacion-de-giroscopio-para-drones/)

---

## Pipeline completo

| Paso | Herramienta | Qué se hace |
|------|-------------|-------------|
| 1. Grabar dataset | `grabar_dataset_v2.py` + RPi Cam v2 | Vídeo del giroscopio en posiciones variadas; extracción de frames |
| 2. Etiquetar | [Roboflow](https://roboflow.com) | 12 keypoints por imagen; augmentaciones permitidas: ruido y blur (flips **prohibidos**) |
| 3. Entrenar | `giroscopio_colab_v4.ipynb` (Google Colab, GPU T4 gratuita) | YOLOv8n-Pose, 150 épocas, exportación a ONNX |
| 4. Compilar HEF | Docker + Hailo DFC 3.33.0 (Ubuntu o Windows) | ONNX → HEF (formato nativo del chip Hailo-8); ~22 min de compilación |
| 5. Inferir | `giroscopio_12kp_v3h.py` en RPi 5 | ~22 FPS · panel visual con arcos, péndulo y brújula · logs automáticos |

---

## Resultados

| Modelo | Imágenes raw | mAP50-95(P) | Conf. media (RPi) | Estado |
|--------|-------------|-------------|-------------------|--------|
| p113 | 113 | 0.130 | ~0.636 | Compilado, no apto producción |
| p211 | 211 | 0.716 | — | Compilado ✓ |
| p309 | 309 | 0.748 | 0.636 | Compilado ✓, probado ✓ |
| **p410** | **410** | **0.764** | **0.888** | **PRODUCCIÓN — 0 pérdidas, 99.5% frames >0.5 conf** |
| a100 *(alumno)* | 100 | 0.130 | 0.737 | Compilado ✓, probado ✓ |

**Conclusión:** El salto crítico de rendimiento está entre 113 y 211 imágenes (+0.586 mAP).
A partir de 211 la curva se aplana — la variedad de fondo importa más que el volumen de imágenes.

**Limitación geométrica documentada:**

| Eje | Fiabilidad | Motivo |
|-----|-----------|--------|
| Pitch | ✅ Fiable | Vista lateral: el plano del octógono cambia sin ambigüedad |
| Roll | ⚠️ Parcial | Vista lateral: indistinguible de cambio en profundidad |
| Yaw | ❌ No funciona | Geométricamente imposible con cámara lateral única |

Esta limitación está documentada y resuelta conceptualmente en el documento principal
(segunda cámara o fusión con IMU MPU-6050).

---

## Estructura del repositorio

```
gyroscope/
│
├── README.md
├── README_EN.md
├── LICENSE
│
├── docs/
│   ├── Proyecto_Giroscopio_12KP_Hailo8_v8.pdf   # Documento completo (112 pág): hardware,
│   │                                              # pipeline, bugs, resultados, pedagogía
│   ├── tutorial_hailo8_ubuntu.pdf                # Compilación ONNX→HEF en Ubuntu con Docker
│   ├── tutorial_hailo8_windows.pdf               # Compilación ONNX→HEF en Windows con Docker
│   └── etiquetas_keypoints.pdf                   # Referencia imprimible de los 12 KPs con colores
│
├── sw/
│   ├── inference/
│   │   ├── giroscopio_12kp_v3h.py                # Script principal de inferencia (producción)
│   │   ├── lanzar_giroscopio.sh                  # Wrapper bash — lanzador sin grabación
│   │   ├── lanzar_giroscopio_video.sh            # Wrapper bash — lanzador con grabación automática
│   │   ├── Giroscopio.desktop                    # Icono de escritorio → lanzar_giroscopio.sh
│   │   ├── Giroscopio_Video.desktop              # Icono de escritorio → lanzar_giroscopio_video.sh
│   │   └── Instalacion_lanzador.txt              # Instrucciones de instalación de los lanzadores
│   ├── dataset/
│   │   └── grabar_dataset_v2.py                  # Grabación de vídeo y extracción de frames
│   └── training/
│       └── giroscopio_colab_v4.ipynb             # Notebook Google Colab — entrenamiento YOLOv8n-Pose
│
├── hailo/
│   └── giroscopio.alls                           # Script de cuantización para Hailo DFC (referencia)
│
├── 3d/                                           # Archivos para impresión 3D (soportes RPi y cámara)
│
└── assets/                                       # Imágenes y GIF para documentación
```

---

## Inicio rápido

### Prerrequisitos

- Raspberry Pi 5 (4 u 8 GB) con Raspberry Pi AI HAT+ (chip Hailo-8)
- Raspberry Pi Camera Module v2
- Raspberry Pi OS Bookworm (64-bit)
- HailoRT 4.23.0 instalado (`hailortcli fw-control identify` debe responder)
- Python 3 con `picamera2`, `opencv-python`, `numpy`

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/nutshell-electronica/gyroscope.git
cd gyroscope

# 2. Crear estructura de carpetas en la RPi
mkdir -p /home/$USER/giroscopio/giro_scripts
mkdir -p /home/$USER/giroscopio/giroscopio_v3a/p410

# 3. Copiar scripts
cp sw/inference/* /home/$USER/giroscopio/giro_scripts/
chmod +x /home/$USER/giroscopio/giro_scripts/*.sh

# 4. Copiar el modelo HEF compilado (ver docs/tutorial_hailo8_ubuntu.pdf para compilarlo)
# scp tu_pc:/ruta/al/giroscopio.hef /home/$USER/giroscopio/giroscopio_v3a/p410/
```

Para instalar los iconos de escritorio, consulta `sw/inference/Instalacion_lanzador.txt`.

### Ejecución

```bash
# Desde terminal
cd /home/$USER/giroscopio/giro_scripts
./lanzar_giroscopio.sh

# Con grabación de vídeo automática
./lanzar_giroscopio_video.sh

# Directamente con parámetros personalizados
python3 giroscopio_12kp_v3h.py \
    --hef /home/$USER/giroscopio/giroscopio_v3a/p410/giroscopio.hef \
    --dead-zone 6 \
    --iou-thresh 0.30
```

### Parámetros configurables

| Parámetro CLI | Valor por defecto | Descripción |
|--------------|-------------------|-------------|
| `--hef` | (ver wrapper .sh) | Ruta al archivo HEF compilado |
| `--dead-zone` | `6.0` | Zona muerta en grados (absorbe ruido de ±5-7°) |
| `--iou-thresh` | `0.30` | Umbral IoU para NMS (bajar si hay detecciones dobles) |
| `--log-dir` | `giro_historico_logs/` | Directorio para logs de sesión |
| `--save-auto` | (desactivado) | Directorio para grabación automática de vídeo |

> ⚠️ Si al lanzar desde el icono de escritorio la ventana no aparece (pantalla negra sin error),
> consulta la sección de lanzadores del documento principal — es el Bug Wayland/XAUTHORITY,
> documentado y resuelto en `Instalacion_lanzador.txt`.

---

## Los 12 keypoints

| Grupo | KPs | Eje | Marcadores de color |
|-------|-----|-----|---------------------|
| OCT (octógono) | KP1–KP4 | Pitch | Blanco, Negro, Rojo, Amarillo |
| INT (soporte U) | KP5–KP8 | Roll | Azul eléctrico, Naranja, Verde lima, Magenta |
| BASE (base cuadrada) | KP9–KP12 | Yaw | Cian, Rosa, Marrón, Morado |

Ver `docs/etiquetas_keypoints.pdf` para la referencia imprimible con posiciones exactas.

> ⚠️ Los flips horizontales y verticales están **prohibidos** en las augmentaciones de Roboflow.
> Invierten el orden de los keypoints simétricos y corrompen las etiquetas.

---

## Documentación completa

📄 **`docs/Proyecto_Giroscopio_12KP_Hailo8_v8.pdf`** — 112 páginas que cubren:

- Diseño y fabricación del giroscopio mecánico
- Arquitectura del sistema de visión artificial
- Pipeline completo paso a paso con capturas reales
- 17 bugs diagnosticados y resueltos durante el desarrollo
- Comparativa de los 5 modelos entrenados
- Limitación geométrica de cámara única (análisis formal)
- Resultados de pruebas físicas con drones
- Reflexión pedagógica y líneas futuras
- Anexos: notebook Colab v4, tutorial Hailo compilación, script de inferencia anotado

---

## Hardware del puesto de trabajo

| Componente | Modelo |
|-----------|--------|
| Computador | Raspberry Pi 5 (8 GB RAM) |
| Acelerador IA | Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS, PCIe) |
| Cámara | Raspberry Pi Camera Module v2 (IMX219, RGB888, 960×540) |
| Monitor | MSI 24,5" 120 Hz |
| Teclado | Logitech K400 inalámbrico |
| Soportes | VESA y cámara impresos en 3D (ver `3d/`) |
| Giroscopio | Modelo comercial ampliado de 3 ejes con 12 marcadores de color |

---

## Contexto educativo

Este proyecto forma parte del **Proyecto de Elaboración de Materiales MTT-041/25**,
concedido por la Consejería de Educación de la Junta de Andalucía con una dotación de 800 €.

**Participantes:** 3 grupos de alumnos de CFGS Mantenimiento Electrónico
(4 alumnos/grupo) del IES Politécnico Jesús Marín (Málaga).
Cada grupo grabó su propio dataset, etiquetó en Roboflow y entrenó su modelo en Google Colab.

Un alumno completó el proceso de forma independiente (modelo a100) y obtuvo resultados
comparables al modelo del profesor, validando que el proceso es **reproducible sin supervisión directa**.

---

## Versión anterior del repositorio

La versión original del proyecto (modelo de 9 keypoints, primer pipeline) está disponible
como [Release v1.0](https://github.com/nutshell-electronica/gyroscope/releases/tag/v1.0).

---

## Licencia

Ver [LICENSE](LICENSE).

---

*IES Politécnico Jesús Marín · Departamento de Electricidad y Electrónica · Málaga · 2026*
