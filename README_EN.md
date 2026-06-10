# Gyroscope Edge AI — YOLOv8n-Pose + Hailo-8 on Raspberry Pi 5

**Real-time orientation detection of a mechanical gyroscope using 12 custom keypoints,
running entirely on-device on a Raspberry Pi 5 with a Hailo-8 NPU — no cloud, no GPU.**

> Educational innovation project MTT-041/25 · IES Politécnico Jesús Marín, Málaga (Spain)  
> Higher Vocational Training in Electronic Maintenance · Drone Maintenance Specialisation  
> Author: José Luis Guerrero Marín · June 2026

---

## What is this?

A computer vision system that detects a physical drone gyroscope and estimates its
orientation angles (Roll, Pitch, Yaw) in real time. A custom YOLOv8n-Pose model detects
12 keypoints distributed across three structural groups of the gyroscope, and computes
3D orientation from their 2D coordinates.

The entire pipeline — capture, inference, computation, and display — runs on a
**Raspberry Pi 5 with AI HAT+ (Hailo-8 chip, 26 TOPS)** with no internet connection and
no external GPU, at ~22 FPS with the production model.

The project has two goals: to serve as a **real hands-on teaching platform** for electronics
students, and to demonstrate that the full Edge AI pipeline
(dataset → labelling → training → compilation → embedded inference)
is **replicable by vocational students** using free tools.

> 🌐 [Versión en español → README.md](README.md)

![Demo](assets/demo.gif)

---

## Demo

> 🎬 *Presentation video — coming soon*

📖 Project blog (Spanish):
[blogsaverroes.juntadeandalucia.es/industria4](https://blogsaverroes.juntadeandalucia.es/industria4/2026/03/15/deteccion-de-orientacion-de-giroscopio-para-drones/)

---

## Full Pipeline

| Step | Tool | What happens |
|------|------|-------------|
| 1. Record dataset | `grabar_dataset_v2.py` + RPi Cam v2 | Video of gyroscope in varied positions; frame extraction |
| 2. Label | [Roboflow](https://roboflow.com) | 12 keypoints per image; allowed augmentations: noise and blur (flips **forbidden**) |
| 3. Train | `giroscopio_colab_v4.ipynb` (Google Colab, free T4 GPU) | YOLOv8n-Pose, 150 epochs, ONNX export |
| 4. Compile HEF | Docker + Hailo DFC 3.33.0 (Ubuntu or Windows) | ONNX → HEF (Hailo-8 native format); ~22 min compile time |
| 5. Infer | `giroscopio_12kp_v3h.py` on RPi 5 | ~22 FPS · visual panel with arcs, pendulum and compass · automatic logs |

---

## Results

| Model | Raw images | mAP50-95(P) | Mean conf (RPi) | Status |
|-------|-----------|-------------|-----------------|--------|
| p113 | 113 | 0.130 | ~0.636 | Compiled, not production-ready |
| p211 | 211 | 0.716 | — | Compiled ✓ |
| p309 | 309 | 0.748 | 0.636 | Compiled ✓, tested ✓ |
| **p410** | **410** | **0.764** | **0.888** | **PRODUCTION — 0 lost frames, 99.5% frames >0.5 conf** |
| a100 *(student)* | 100 | 0.130 | 0.737 | Compiled ✓, tested ✓ |

**Key finding:** The critical performance jump is between 113 and 211 images (+0.586 mAP).
Beyond 211, the curve flattens — background variety matters more than image volume.

**Documented geometric limitation:**

| Axis | Reliability | Reason |
|------|------------|--------|
| Pitch | ✅ Reliable | Side view: octagon plane angle changes unambiguously |
| Roll | ⚠️ Partial | Side view: indistinguishable from depth change |
| Yaw | ❌ Not working | Geometrically impossible with a single lateral camera |

This limitation is formally documented and conceptually resolved in the main document
(second camera or IMU MPU-6050 fusion).

---

## Repository Structure

```
gyroscope/
│
├── README.md                                      # Spanish version
├── README_EN.md                                   # This file
├── LICENSE
│
├── docs/
│   ├── Proyecto_Giroscopio_12KP_Hailo8_v8.pdf   # Full document (112 pp): hardware,
│   │                                              # pipeline, bugs, results, pedagogy
│   ├── tutorial_hailo8_ubuntu.pdf                # ONNX→HEF compilation on Ubuntu with Docker
│   ├── tutorial_hailo8_windows.pdf               # ONNX→HEF compilation on Windows with Docker
│   └── etiquetas_keypoints.pdf                   # Printable 12-KP reference with colours
│
├── sw/
│   ├── inference/
│   │   ├── giroscopio_12kp_v3h.py                # Main inference script (production)
│   │   ├── lanzar_giroscopio.sh                  # Bash wrapper — launch without recording
│   │   ├── lanzar_giroscopio_video.sh            # Bash wrapper — launch with auto video recording
│   │   ├── Giroscopio.desktop                    # Desktop icon → lanzar_giroscopio.sh
│   │   ├── Giroscopio_Video.desktop              # Desktop icon → lanzar_giroscopio_video.sh
│   │   └── Instalacion_lanzador.txt              # Launcher installation instructions
│   ├── dataset/
│   │   └── grabar_dataset_v2.py                  # Video recording and frame extraction
│   └── training/
│       └── giroscopio_colab_v4.ipynb             # Google Colab notebook — YOLOv8n-Pose training
│
├── hailo/
│   └── giroscopio.alls                           # Hailo DFC quantisation script (reference)
│
├── 3d/                                           # 3D printable files (RPi and camera mounts)
│
└── assets/                                       # Images and GIF for documentation
```

---

## Quick Start

### Prerequisites

- Raspberry Pi 5 (4 or 8 GB) with Raspberry Pi AI HAT+ (Hailo-8 chip)
- Raspberry Pi Camera Module v2
- Raspberry Pi OS Bookworm (64-bit)
- HailoRT 4.23.0 installed (`hailortcli fw-control identify` should respond)
- Python 3 with `picamera2`, `opencv-python`, `numpy`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nutshell-electronica/gyroscope.git
cd gyroscope

# 2. Create folder structure on the RPi
mkdir -p /home/$USER/giroscopio/giro_scripts
mkdir -p /home/$USER/giroscopio/giroscopio_v3a/p410

# 3. Copy scripts
cp sw/inference/* /home/$USER/giroscopio/giro_scripts/
chmod +x /home/$USER/giroscopio/giro_scripts/*.sh

# 4. Copy the compiled HEF model (see docs/tutorial_hailo8_ubuntu.pdf to compile it)
# scp your_pc:/path/to/giroscopio.hef /home/$USER/giroscopio/giroscopio_v3a/p410/
```

For desktop icon installation, see `sw/inference/Instalacion_lanzador.txt`.

### Running

```bash
# From terminal
cd /home/$USER/giroscopio/giro_scripts
./lanzar_giroscopio.sh

# With automatic video recording
./lanzar_giroscopio_video.sh

# Directly with custom parameters
python3 giroscopio_12kp_v3h.py \
    --hef /home/$USER/giroscopio/giroscopio_v3a/p410/giroscopio.hef \
    --dead-zone 6 \
    --iou-thresh 0.30
```

### Configurable Parameters

| CLI parameter | Default | Description |
|--------------|---------|-------------|
| `--hef` | (see .sh wrapper) | Path to compiled HEF file |
| `--dead-zone` | `6.0` | Dead zone in degrees (absorbs ±5–7° noise) |
| `--iou-thresh` | `0.30` | IoU threshold for NMS (lower if double detections appear) |
| `--log-dir` | `giro_historico_logs/` | Directory for session logs |
| `--save-auto` | (disabled) | Directory for automatic video recording |

> ⚠️ If the window does not appear when launching from the desktop icon (silent black screen),
> see the launcher section of the main document — this is the Wayland/XAUTHORITY bug,
> documented and resolved in `Instalacion_lanzador.txt`.

---

## The 12 Keypoints

| Group | KPs | Axis | Colour markers |
|-------|-----|------|---------------|
| OCT (octagon) | KP1–KP4 | Pitch | White, Black, Red, Yellow |
| INT (U-support) | KP5–KP8 | Roll | Electric blue, Orange, Lime green, Magenta |
| BASE (square base) | KP9–KP12 | Yaw | Cyan, Pink, Brown, Purple |

See `docs/etiquetas_keypoints.pdf` for the printable reference with exact positions.

> ⚠️ Horizontal and vertical flips are **forbidden** in Roboflow augmentations.
> They reverse the keypoint label order of symmetric points and corrupt annotations.

---

## Full Documentation

📄 **`docs/Proyecto_Giroscopio_12KP_Hailo8_v8.pdf`** — 112 pages covering:

- Mechanical gyroscope design and fabrication
- Computer vision system architecture
- Full pipeline walkthrough with real screenshots
- 17 bugs diagnosed and resolved during development
- Comparison of all 5 trained models
- Single-camera geometric limitation (formal analysis)
- Physical test results with drones
- Pedagogical reflection and future directions
- Appendices: Colab notebook v4, Hailo compilation tutorial, annotated inference script

---

## Hardware Setup

| Component | Model |
|-----------|-------|
| Computer | Raspberry Pi 5 (8 GB RAM) |
| AI accelerator | Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS, PCIe) |
| Camera | Raspberry Pi Camera Module v2 (IMX219, RGB888, 960×540) |
| Monitor | MSI 24.5" 120 Hz |
| Keyboard | Logitech K400 wireless |
| Mounts | 3D-printed VESA and camera mounts (see `3d/`) |
| Gyroscope | Commercial 3-axis model with 12 colour markers |

---

## Educational Context

This project is part of the **Teaching Materials Project MTT-041/25**,
funded by the Andalusian Regional Government (Junta de Andalucía) with an 800 € grant.

**Participants:** 3 groups of Higher Vocational Training students in Electronic Maintenance
(4 students/group) at IES Politécnico Jesús Marín (Málaga, Spain).
Each group recorded their own dataset, labelled it in Roboflow, and trained their model in Google Colab.

One student completed the process independently (model a100) and achieved results
comparable to the teacher's model, validating that the pipeline is **reproducible without direct supervision**.

---

## Previous Version

The original version of the project (9-keypoint model, first pipeline) is available
as [Release v1.0](https://github.com/nutshell-electronica/gyroscope/releases/tag/v1.0).

---

## License

See [LICENSE](LICENSE).

---

*IES Politécnico Jesús Marín · Department of Electrical & Electronic Engineering · Málaga · 2026*
