#!/bin/bash
# ============================================================
# lanzar_giroscopio_video.sh — Lanzador con grabación de vídeo
#
# PROPÓSITO:
#   Igual que lanzar_giroscopio.sh pero activa la grabación
#   automática de vídeo con nombre estandarizado.
#
# NOMENCLATURA DE VÍDEO:
#   giroscopio_v3a_<modelo>_<YYYYMMDD_HHMMSS>.mp4
#   Ejemplo: giroscopio_v3a_p410_20260608_143022.mp4
#   El <modelo> lo extrae el script Python del directorio padre del HEF.
#
# NOMENCLATURA DE LOGS (igual que el lanzador normal):
#   logs_giroscopio_v3a_<modelo>_<YYYYMMDD_HHMMSS>.txt
#
# ESTRUCTURA DE DIRECTORIOS ESPERADA:
#   /home/ai/giroscopio/
#   ├── giro_scripts/
#   │   ├── giroscopio_12kp_v3h.py
#   │   ├── lanzar_giroscopio.sh
#   │   └── lanzar_giroscopio_video.sh    ← este archivo
#   ├── giro_historico_logs/              ← logs aquí
#   ├── giro_resultados_videos/           ← videos aquí
#   └── giroscopio_v3a/
#       └── p410/
#           └── giroscopio.hef
#
# USO DESDE TERMINAL (acepta flags extra para el script Python):
#   ./lanzar_giroscopio_video.sh
#   ./lanzar_giroscopio_video.sh --hide-yaw
#
# VERSIÓN: v1.0 — 08/06/2026
# PROYECTO: MTT-041/25 — IES Politécnico Jesús Marín
# ============================================================

# ─── PARÁMETROS CONFIGURABLES ────────────────────────────────
SCRIPT_DIR="/home/ai/giroscopio/giro_scripts"
SCRIPT_NAME="giroscopio_12kp_v3h.py"
HEF_PATH="/home/ai/giroscopio/giroscopio_v3a/p410/giroscopio.hef"
LOG_DIR="/home/ai/giroscopio/giro_historico_logs"
VIDEO_DIR="/home/ai/giroscopio/giro_resultados_videos"
PYTHON_BIN="python3"
# VENV_ACTIVATE="$SCRIPT_DIR/venv/bin/activate"   # Descomentar si usas venv
# ─────────────────────────────────────────────────────────────

# --- Forzar X11 (necesario en Bookworm/Wayland para cv2.imshow) ---
export DISPLAY=:0
unset WAYLAND_DISPLAY

echo "=================================================="
echo "  GIROSCOPIO 12KP — Lanzador con vídeo v1.0"
echo "  $(date)"
echo "  Script : $SCRIPT_DIR/$SCRIPT_NAME"
echo "  HEF    : $HEF_PATH"
echo "  Logs   : $LOG_DIR"
echo "  Videos : $VIDEO_DIR"
echo "  Nombre : giroscopio_v3a_<modelo>_<fecha_hora>.mp4"
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

# Crear directorios si no existen
mkdir -p "$LOG_DIR"
mkdir -p "$VIDEO_DIR"

# --- Lanzar desde el directorio del script ---
cd "$SCRIPT_DIR"
echo "[INFO] Ejecutando:"
echo "       $PYTHON_BIN $SCRIPT_NAME --hef $HEF_PATH"
echo "           --log-dir $LOG_DIR"
echo "           --save-auto $VIDEO_DIR $@"
echo "--------------------------------------------------"

$PYTHON_BIN "$SCRIPT_NAME" \
    --hef "$HEF_PATH" \
    --log-dir "$LOG_DIR" \
    --save-auto "$VIDEO_DIR" \
    "$@"
EXIT_CODE=$?

echo "--------------------------------------------------"
echo "[INFO] Terminado con código: $EXIT_CODE"
[ $EXIT_CODE -ne 0 ] && echo "[AVISO] Revisa los mensajes anteriores."

# Mostrar el último vídeo generado para confirmación
ULTIMO_VIDEO=$(ls -t "$VIDEO_DIR"/giroscopio_v3a_*.mp4 2>/dev/null | head -1)
if [ -n "$ULTIMO_VIDEO" ]; then
    echo "[INFO] Vídeo guardado en: $ULTIMO_VIDEO"
    SIZE=$(du -sh "$ULTIMO_VIDEO" 2>/dev/null | cut -f1)
    echo "[INFO] Tamaño: $SIZE"
fi

read -p "Pulsa ENTER para cerrar..."
