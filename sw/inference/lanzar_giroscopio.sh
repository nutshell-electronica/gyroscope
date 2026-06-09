#!/bin/bash
# ============================================================
# lanzar_giroscopio.sh — Lanzador normal (solo visualización)
#
# PROPÓSITO:
#   Lanza giroscopio_12kp_v3h.py desde el escritorio de
#   Raspberry Pi OS Bookworm con un doble clic.
#   Los logs se guardan en giro_historico_logs/ (un nivel
#   arriba de giro_scripts/).
#
# ESTRUCTURA DE DIRECTORIOS ESPERADA:
#   /home/ai/giroscopio/
#   ├── giro_scripts/
#   │   ├── giroscopio_12kp_v3h.py
#   │   ├── lanzar_giroscopio.sh          ← este archivo
#   │   └── lanzar_giroscopio_video.sh
#   ├── giro_historico_logs/              ← logs aquí
#   ├── giro_resultados_videos/           ← videos aquí
#   └── giroscopio_v3a/
#       └── p410/
#           └── giroscopio.hef
#
# NOMENCLATURA DE LOGS:
#   logs_giroscopio_v3a_<modelo>_<YYYYMMDD_HHMMSS>.txt
#   Ejemplo: logs_giroscopio_v3a_p410_20260608_143022.txt
#
# USO DESDE TERMINAL (acepta flags extra para el script Python):
#   ./lanzar_giroscopio.sh
#   ./lanzar_giroscopio.sh --hide-yaw --iou-thresh 0.25
#   ./lanzar_giroscopio.sh --verbose
#
# VERSIÓN: v1.2 — 08/06/2026
# PROYECTO: MTT-041/25 — IES Politécnico Jesús Marín
# ============================================================

# ─── PARÁMETROS CONFIGURABLES ────────────────────────────────
SCRIPT_DIR="/home/ai/giroscopio/giro_scripts"
SCRIPT_NAME="giroscopio_12kp_v3h.py"
HEF_PATH="/home/ai/giroscopio/giroscopio_v3a/p410/giroscopio.hef"
LOG_DIR="/home/ai/giroscopio/giro_historico_logs"
PYTHON_BIN="python3"
# VENV_ACTIVATE="$SCRIPT_DIR/venv/bin/activate"   # Descomentar si usas venv
# ─────────────────────────────────────────────────────────────

# --- Forzar X11 (necesario en Bookworm/Wayland para cv2.imshow) ---
export DISPLAY=:0
unset WAYLAND_DISPLAY

echo "=================================================="
echo "  GIROSCOPIO 12KP — Lanzador normal v1.2"
echo "  $(date)"
echo "  Script : $SCRIPT_DIR/$SCRIPT_NAME"
echo "  HEF    : $HEF_PATH"
echo "  Logs   : $LOG_DIR"
echo "=================================================="

# --- Activar venv si está configurado ---
if [ -n "$VENV_ACTIVATE" ] && [ -f "$VENV_ACTIVATE" ]; then
    echo "[INFO] Activando entorno virtual: $VENV_ACTIVATE"
    source "$VENV_ACTIVATE"
else
    echo "[INFO] Sin entorno virtual — usando Python del sistema"
fi

# --- Verificaciones previas ---
ERRORES=0
[ ! -f "$SCRIPT_DIR/$SCRIPT_NAME" ] && \
    echo "[ERROR] Script no encontrado: $SCRIPT_DIR/$SCRIPT_NAME" && ERRORES=1
[ ! -f "$HEF_PATH" ] && \
    echo "[ERROR] HEF no encontrado: $HEF_PATH" && ERRORES=1

if [ $ERRORES -ne 0 ]; then
    echo "--------------------------------------------------"
    echo "[FATAL] Corrige los errores anteriores."
    read -p "Pulsa ENTER para cerrar..."
    exit 1
fi

# Crear directorio de logs si no existe
mkdir -p "$LOG_DIR"

# --- Lanzar desde el directorio del script ---
cd "$SCRIPT_DIR"
echo "[INFO] Ejecutando:"
echo "       $PYTHON_BIN $SCRIPT_NAME --hef $HEF_PATH --log-dir $LOG_DIR $@"
echo "--------------------------------------------------"

$PYTHON_BIN "$SCRIPT_NAME" \
    --hef "$HEF_PATH" \
    --log-dir "$LOG_DIR" \
    "$@"
EXIT_CODE=$?

echo "--------------------------------------------------"
echo "[INFO] Terminado con código: $EXIT_CODE"
[ $EXIT_CODE -ne 0 ] && echo "[AVISO] Revisa los mensajes anteriores."
read -p "Pulsa ENTER para cerrar..."
