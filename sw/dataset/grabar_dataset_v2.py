#!/usr/bin/env python3
"""
grabar_dataset_v2.py — Grabación de vídeo y extracción de imágenes para dataset
================================================================================
Proyecto de innovación educativa — IES Politécnico Jesús Marín, Málaga
Familia Electricidad y Electrónica — Junta de Andalucía

PROPÓSITO
---------
Grabar un vídeo con la Raspberry Pi Camera v2 (IMX219) y, al finalizar,
extraer automáticamente dos conjuntos de imágenes para:
  - Entrenamiento del modelo YOLOv8n-Pose en Google Colab
  - Calibración del HEF durante la compilación con Hailo DFC

LÓGICA DE EXTRACCIÓN
---------------------
Se extrae 1 frame cada INTERVALO_SEGUNDOS segundos del vídeo grabado.
Los frames se asignan de forma intercalada:
  frame par  → carpeta entrenamiento/
  frame impar → carpeta calibracion/

Ejemplos de duración mínima para obtener N imágenes en total (N/2 por carpeta):
  --num-imagenes 200 --intervalo 0.5 → mínimo  100s (~2 min)
  --num-imagenes 200 --intervalo 1.5 → mínimo  300s (~5 min)
  --num-imagenes 600 --intervalo 1.0 → mínimo  600s (~10 min)
  --num-imagenes 600 --intervalo 2.0 → mínimo 1200s (~20 min)

Con menos tiempo el modo reparto equitativo distribuye lo disponible
a partes iguales entre entrenamiento y calibración.

CONTADOR DE FRAMES EN TIEMPO REAL (OSD durante la grabación)
-------------------------------------------------------------
El OSD muestra Train: NNN/300 y Calib: NNN/300 actualizándose en
pantalla a medida que transcurre el tiempo, sin esperar a la
extracción posterior. También muestra una mini-barra de cuenta atrás
hasta el próximo frame y el intervalo configurado (@X.Xs).
El contador es una estimación basada en el tiempo; la extracción real
puede diferir en ±1 frame por redondeos de FPS.

FORMATO DE IMAGEN
-----------------
Las imágenes se guardan en JPEG en RGB888 (formato nativo Picamera2),
SIN conversión a BGR. Coherente con CONVERT_TO_BGR=False en v3b.

VISUALIZACIÓN DURANTE LA GRABACIÓN
------------------------------------
Se muestra una ventana OpenCV con el frame en tiempo real.
El OSD (tiempo, barra de progreso, tecla Q) se superpone SOLO en la ventana
— NO se escribe en el vídeo ni en las imágenes extraídas.
Pulsar Q o cerrar la ventana corta la grabación y pasa a la extracción
con el material grabado hasta ese momento.

ESTRUCTURA DE SALIDA
--------------------
  RUTA_BASE/
    giroscopio_YYYYMMDD_HHMMSS.mp4    ← vídeo completo
    entrenamiento/
      frame_000000.jpg  (frames pares del vídeo)
    calibracion/
      frame_000001.jpg  (frames impares del vídeo)
    resumen_extraccion.txt

CÓMO USAR
----------
  # 200 imágenes totales (100 train + 100 calib), 1 frame cada 1.5s:
  DISPLAY=:0 python3 grabar_dataset_v2.py --num-imagenes 200 --intervalo 1.5

  # Especificar también duración máxima:
  DISPLAY=:0 python3 grabar_dataset_v2.py --num-imagenes 200 --intervalo 1.5 --duracion 480

  # Directorio de salida personalizado:
  DISPLAY=:0 python3 grabar_dataset_v2.py --num-imagenes 200 --salida /home/ai/dataset_escena2

  # Solo extraer frames de un vídeo ya grabado (sin grabar):
  python3 grabar_dataset_v2.py --solo-extraer /home/ai/mi_video.mp4 --num-imagenes 200

  # Sin ventana de preview:
  python3 grabar_dataset_v2.py --num-imagenes 200 --sin-preview

  # Compatibilidad v1 (--objetivo = imágenes POR carpeta):
  python3 grabar_dataset_v2.py --objetivo 100   # equivale a --num-imagenes 200

NOTAS IMPORTANTES PARA EL ETIQUETADO POSTERIOR
-----------------------------------------------
1. Usar Roboflow para etiquetar las imágenes de entrenamiento/.
2. La carpeta calibracion/ NO se etiqueta — se usa tal cual en el Hailo DFC.
3. El giroscopio debe ocupar el 40-70% del frame durante la grabación.
4. Marcar KPs ocultos como "occluded" en Roboflow — nunca omitirlos.
5. NO usar flips como augmentation — pueden invertir el orden de KPs.
   Sí se pueden usar brillo, contraste, ruido gaussiano.

HISTORIA DE VERSIONES
---------------------
v1 — 21/05/2026
  Script inicial. Grabación a ciegas, Ctrl+C para detener.

v1.1 — 21/05/2026
  NUEVO: ventana de preview en tiempo real con OSD superpuesto.
  NUEVO: contador de tiempo transcurrido y tiempo restante en pantalla.
  NUEVO: barra de progreso visual en la ventana.
  NUEVO: detener la grabación pulsando Q (además de Ctrl+C).
  NUEVO: flag --sin-preview para entornos sin pantalla.
  El OSD solo aparece en la ventana, NO se escribe en el vídeo ni
  en las imágenes que se extraen para el dataset.

v2 — 22/05/2026
  NUEVO: --num-imagenes N indica el TOTAL de imágenes (train + calib).
    Reparto automático 50/50. Más intuitivo que --objetivo (por carpeta).
    Ejemplo: --num-imagenes 200 → 100 train + 100 calib.
  NUEVO: validación al arrancar — calcula y muestra la duración mínima
    necesaria según num_imagenes e intervalo, con aviso si no alcanza.
  COMPATIBILIDAD: --objetivo sigue funcionando como en v1 (imágenes por
    carpeta). Si se pasan los dos, --num-imagenes tiene prioridad.
  SIN CAMBIOS en: lógica de grabación, OSD, extracción, modo reparto.
"""

import argparse
import datetime
import os
import sys
import time

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    HAVE_PICAMERA = True
except ImportError:
    HAVE_PICAMERA = False

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — ajustar según necesidad
# ════════════════════════════════════════════════════════════════════════════

# Resolución de grabación — igual que la de inferencia para máxima coherencia
CAM_W, CAM_H = 960, 540

# Duración mínima recomendada: 300 s (5 min).
DURACION_SEGUNDOS_DEFAULT = 300

# Intervalo entre frames extraídos (segundos).
# 0.5 s → 2 frames/s → con 5 min → 600 frames → 300 + 300
INTERVALO_SEGUNDOS_DEFAULT = 0.5

# Número objetivo de imágenes POR carpeta (alias v1)
# v2: usa --num-imagenes (total) en CLI; este valor es el fallback
OBJETIVO_IMAGENES = 300

# FPS de grabación del vídeo
FPS_GRABACION = 30

# Calidad JPEG de las imágenes extraídas (0-100)
JPEG_CALIDAD = 95

# Directorio base de salida
RUTA_BASE_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# ── OSD (On-Screen Display) — solo afecta a la ventana, nunca al vídeo ────
# Fuente OpenCV para el OSD
OSD_FONT       = cv2.FONT_HERSHEY_SIMPLEX
OSD_COLOR_OK   = (0,   220,   0)    # verde  — tiempo transcurrido, barra
OSD_COLOR_WARN = (0,   200, 220)    # amarillo — tiempo restante
OSD_COLOR_BG   = (20,   20,  20)    # fondo semitransparente del panel
OSD_COLOR_Q    = (60,  180, 255)    # naranja — texto tecla Q
OSD_COLOR_REC  = (0,     0, 220)    # rojo   — indicador REC

# ════════════════════════════════════════════════════════════════════════════
#  ARGUMENTOS CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Graba vídeo con RPi Camera y extrae imágenes para dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  DISPLAY=:0 python3 grabar_dataset_v2.py --num-imagenes 200 --intervalo 1.5
  DISPLAY=:0 python3 grabar_dataset_v2.py --num-imagenes 200 --duracion 480
  DISPLAY=:0 python3 grabar_dataset_v2.py --salida /home/ai/dataset_escena2
  python3 grabar_dataset_v2.py --solo-extraer /home/ai/video.mp4 --num-imagenes 200
  python3 grabar_dataset_v2.py --sin-preview --num-imagenes 200
        """)

    # ── NUEVO en v2: parámetro principal (total train+calib) ─────────────────
    p.add_argument("--num-imagenes", type=int,   default=None,
                   metavar="N",
                   help="Nº TOTAL de imágenes a extraer (train+calib). "
                        "Reparto 50/50. Ej: --num-imagenes 200 → 100+100. "
                        f"Default: {OBJETIVO_IMAGENES * 2}")

    # ── Alias de compatibilidad con v1 (imágenes POR carpeta) ────────────────
    p.add_argument("--objetivo",     type=int,   default=None,
                   metavar="N",
                   help=f"[v1 deprecated] Nº de imágenes POR carpeta "
                        f"(equivale a --num-imagenes N*2). "
                        f"Default si no se especifica ninguno: {OBJETIVO_IMAGENES}")

    p.add_argument("--duracion",     type=int,   default=DURACION_SEGUNDOS_DEFAULT,
                   help=f"Duración en segundos (default: {DURACION_SEGUNDOS_DEFAULT})")
    p.add_argument("--intervalo",    type=float, default=INTERVALO_SEGUNDOS_DEFAULT,
                   help=f"Intervalo entre frames extraídos, en segundos (default: {INTERVALO_SEGUNDOS_DEFAULT})")
    p.add_argument("--salida",       type=str,   default=RUTA_BASE_DEFAULT,
                   help=f"Directorio base de salida (default: {RUTA_BASE_DEFAULT})")
    p.add_argument("--solo-extraer", type=str,   default="",
                   help="Ruta a un MP4 ya grabado: solo extrae frames sin grabar")
    p.add_argument("--sin-preview",  action="store_true",
                   help="No abrir ventana de preview (útil sin pantalla conectada)")

    args = p.parse_args()

    # ── Resolver --num-imagenes vs --objetivo ─────────────────────────────────
    if args.num_imagenes is not None and args.objetivo is not None:
        # Los dos especificados: --num-imagenes gana
        print(f"[AVISO] Se especificaron --num-imagenes y --objetivo a la vez. "
              f"--num-imagenes {args.num_imagenes} tiene prioridad.")
    elif args.num_imagenes is None and args.objetivo is not None:
        # Solo --objetivo (v1): convertir a num_imagenes total
        args.num_imagenes = args.objetivo * 2
        print(f"[AVISO] --objetivo {args.objetivo} (v1 deprecated) → "
              f"--num-imagenes {args.num_imagenes}. "
              f"Usa --num-imagenes en el futuro.")
    elif args.num_imagenes is None:
        # Ninguno: usar el default
        args.num_imagenes = OBJETIVO_IMAGENES * 2

    # Forzar par para reparto 50/50 exacto
    if args.num_imagenes % 2 != 0:
        args.num_imagenes += 1
        print(f"[AVISO] --num-imagenes ajustado a {args.num_imagenes} (debe ser par).")

    # Calcular objetivo por carpeta y guardarlo en args para comodidad
    args.objetivo_carpeta = args.num_imagenes // 2

    return args

# ════════════════════════════════════════════════════════════════════════════
#  OSD — dibuja el panel de información en el frame de preview
#  Se llama con una COPIA del frame — el original que se graba no se toca.
# ════════════════════════════════════════════════════════════════════════════

def dibujar_osd(frame_display, elapsed, duracion_s,
                n_train, n_calib, objetivo, intervalo_s,
                proximo_en):
    """
    Superpone sobre frame_display (BGR) el panel de información de grabación.
    Modifica frame_display IN PLACE. Llamar siempre con una COPIA del frame.

    Parámetros:
      elapsed     — segundos transcurridos desde el inicio
      duracion_s  — duración máxima configurada
      n_train     — imágenes de entrenamiento capturadas hasta ahora
      n_calib     — imágenes de calibración capturadas hasta ahora
      objetivo    — objetivo por carpeta
      intervalo_s — intervalo entre frames (segundos), para mostrarlo
      proximo_en  — segundos hasta el próximo frame (para la barra de cuenta atrás)

    Layout del panel (esquina superior izquierda):
    ┌─────────────────────────────────────────┐
    │ ● REC   MM:SS    -MM:SS                 │  fila 1: tiempo
    │ [████████████████░░░░░░░░] progreso      │  fila 2: barra tiempo
    │ Train: NNN/OOO   Calib: NNN/OOO         │  fila 3: contadores frames
    │ [████░░░░░░] próximo frame en X.Xs       │  fila 4: cuenta atrás frame
    └─────────────────────────────────────────┘
    Q — detener grabacion                        esquina inferior izq
    """
    h, w   = frame_display.shape[:2]
    restante = max(0.0, duracion_s - elapsed)
    progreso = min(1.0, elapsed / duracion_s) if duracion_s > 0 else 0.0

    panel_h = 148
    panel_w = 420
    overlay = frame_display.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), OSD_COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.62, frame_display, 0.38, 0, frame_display)

    # ── Fila 1: REC + tiempo transcurrido + tiempo restante ──────────────────
    if int(elapsed) % 2 == 0:
        cv2.circle(frame_display, (18, 18), 8, OSD_COLOR_REC, -1)
    cv2.putText(frame_display, "REC", (30, 24),
                OSD_FONT, 0.48, OSD_COLOR_REC, 1)

    mins_e, secs_e = divmod(int(elapsed), 60)
    mins_r, secs_r = divmod(int(restante), 60)
    cv2.putText(frame_display, f"{mins_e:02d}:{secs_e:02d}",
                (72, 26), OSD_FONT, 0.90, OSD_COLOR_OK, 2)
    cv2.putText(frame_display, f"-{mins_r:02d}:{secs_r:02d}",
                (180, 26), OSD_FONT, 0.65, OSD_COLOR_WARN, 2)

    # ── Fila 2: barra de progreso de tiempo ──────────────────────────────────
    bx, by, bw, bh = 10, 36, panel_w - 20, 10
    cv2.rectangle(frame_display, (bx, by), (bx + bw, by + bh), (55, 55, 55), -1)
    fill = int(bw * progreso)
    if fill > 0:
        cv2.rectangle(frame_display, (bx, by), (bx + fill, by + bh), OSD_COLOR_OK, -1)
    cv2.rectangle(frame_display, (bx, by), (bx + bw, by + bh), (110, 110, 110), 1)

    # ── Fila 3: contadores de frames por carpeta ──────────────────────────────
    # Color de cada contador: verde si aún no llegó al objetivo, blanco si lo alcanzó
    col_train = (255, 255, 255) if n_train >= objetivo else OSD_COLOR_OK
    col_calib = (255, 255, 255) if n_calib >= objetivo else (100, 220, 255)
    cv2.putText(frame_display, f"Train: {n_train:3d}/{objetivo}",
                (10, 72), OSD_FONT, 0.60, col_train, 2)
    cv2.putText(frame_display, f"Calib: {n_calib:3d}/{objetivo}",
                (220, 72), OSD_FONT, 0.60, col_calib, 2)

    # ── Fila 4: cuenta atrás hasta el próximo frame + intervalo configurado ───
    # La barra se vacía y se rellena cada `intervalo_s` segundos.
    # Cuando llega a cero (o pasa), el script captura el siguiente frame.
    ratio_proximo = 1.0 - min(1.0, proximo_en / intervalo_s) if intervalo_s > 0 else 1.0
    bx2, by2 = 10, 88
    bw2, bh2 = panel_w - 130, 10
    cv2.rectangle(frame_display, (bx2, by2), (bx2 + bw2, by2 + bh2), (55, 55, 55), -1)
    fill2 = int(bw2 * ratio_proximo)
    if fill2 > 0:
        # La barra se pone blanca en el instante de captura para feedback visual
        col_barra = (220, 220, 80) if proximo_en > 0.15 else (255, 255, 255)
        cv2.rectangle(frame_display, (bx2, by2), (bx2 + fill2, by2 + bh2), col_barra, -1)
    cv2.rectangle(frame_display, (bx2, by2), (bx2 + bw2, by2 + bh2), (110, 110, 110), 1)
    cv2.putText(frame_display, f"@{intervalo_s:.1f}s  {proximo_en:.1f}s",
                (bx2 + bw2 + 8, by2 + 10), OSD_FONT, 0.48, (160, 160, 160), 1)

    # ── Total de frames acumulados ────────────────────────────────────────────
    total = n_train + n_calib
    cv2.putText(frame_display, f"Total: {total}  ({intervalo_s:.1f}s/frame)",
                (10, 122), OSD_FONT, 0.45, (160, 160, 160), 1)

    # ── Aviso de objetivo alcanzado ───────────────────────────────────────────
    if n_train >= objetivo and n_calib >= objetivo:
        cv2.putText(frame_display, "OBJETIVO ALCANZADO — puedes parar con Q",
                    (10, 142), OSD_FONT, 0.45, (0, 255, 255), 1)

    # ── Tecla Q ───────────────────────────────────────────────────────────────
    cv2.putText(frame_display, "Q — detener grabacion",
                (10, h - 12), OSD_FONT, 0.50, OSD_COLOR_Q, 1)

# ════════════════════════════════════════════════════════════════════════════
#  GRABACIÓN DE VÍDEO CON PICAMERA2
# ════════════════════════════════════════════════════════════════════════════

def grabar_video(ruta_mp4, duracion_s, intervalo_s, objetivo, log_lines, mostrar_preview):
    """
    Graba vídeo con Picamera2 durante duracion_s segundos.

    Durante la grabación lleva la cuenta de cuántos frames de dataset se han
    "capturado" virtualmente según el intervalo configurado, de modo que el OSD
    muestra en tiempo real el contador Train/Calib sin esperar a la extracción.
    Los frames reales se extraen del vídeo al finalizar; estos contadores son
    una estimación basada en el tiempo transcurrido y el intervalo.

    La extracción real posterior puede diferir en ±1 frame por redondeos de FPS.
    """
    if not HAVE_PICAMERA:
        msg = "ERROR: Picamera2 no disponible. Ejecutar en Raspberry Pi."
        print(msg); log_lines.append(msg)
        return False

    print(f"Iniciando grabación: {duracion_s}s → {ruta_mp4}")
    log_lines.append(f"Grabación: {duracion_s}s → {ruta_mp4}")

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"}))
    cam.start()
    time.sleep(2.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(ruta_mp4, fourcc, FPS_GRABACION, (CAM_W, CAM_H))
    if not writer.isOpened():
        msg = f"ERROR: No se pudo abrir VideoWriter en {ruta_mp4}"
        print(msg); log_lines.append(msg)
        cam.stop()
        return False

    win_title = "Grabacion dataset — Q para detener"
    if mostrar_preview:
        os.environ.setdefault("DISPLAY", ":0")
        os.environ.pop("WAYLAND_DISPLAY", None)
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, CAM_W * 2, CAM_H * 2)

    t_inicio        = time.time()
    frames_grabados = 0
    detenido_por_q  = False

    # ── Estado del contador de frames en tiempo real ──────────────────────────
    # Llevamos la cuenta de cuántos "slots" de intervalo han pasado desde el
    # inicio. Cada slot par va a Train, cada slot impar va a Calib.
    # t_ultimo_frame: marca de tiempo del último slot capturado.
    t_ultimo_frame  = t_inicio   # el primer frame se captura en t=0 (slot 0)
    n_train_est     = 0          # estimación Train
    n_calib_est     = 0          # estimación Calib
    slot_idx        = 0          # índice del próximo slot (0, 1, 2, ...)

    print("Grabando... (pulsa Q en la ventana para detener)")

    try:
        while True:
            elapsed = time.time() - t_inicio
            if elapsed >= duracion_s:
                break

            frame_rgb = cam.capture_array()
            writer.write(frame_rgb)
            frames_grabados += 1

            now = time.time()

            # ── Actualizar contador de slots de intervalo ─────────────────────
            # Mientras hayan pasado más de `intervalo_s` desde el último slot,
            # incrementar el contador correspondiente.
            while now - t_ultimo_frame >= intervalo_s:
                if slot_idx % 2 == 0:
                    n_train_est = min(n_train_est + 1, objetivo)
                else:
                    n_calib_est = min(n_calib_est + 1, objetivo)
                t_ultimo_frame += intervalo_s
                slot_idx       += 1

            # Segundos hasta el próximo slot (para la barra de cuenta atrás)
            proximo_en = max(0.0, t_ultimo_frame + intervalo_s - now)

            if mostrar_preview:
                frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                dibujar_osd(frame_display, elapsed, duracion_s,
                            n_train_est, n_calib_est, objetivo,
                            intervalo_s, proximo_en)
                cv2.imshow(win_title, frame_display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    detenido_por_q = True
                    break
                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                    detenido_por_q = True
                    break

    except KeyboardInterrupt:
        elapsed_real = time.time() - t_inicio
        msg = f"Grabación interrumpida por Ctrl+C a los {elapsed_real:.1f}s"
        print(f"\n{msg}")
        log_lines.append(msg)

    finally:
        cam.stop()
        writer.release()
        if mostrar_preview:
            cv2.destroyAllWindows()

    elapsed_final = time.time() - t_inicio
    fps_media     = frames_grabados / elapsed_final if elapsed_final > 0 else 0
    modo_fin      = "detenida con Q" if detenido_por_q else "completada"
    msg = (f"Grabación {modo_fin}: {elapsed_final:.1f}s  "
           f"frames={frames_grabados}  fps_media={fps_media:.1f}  "
           f"estimado train≈{n_train_est} calib≈{n_calib_est}")
    print(msg)
    log_lines.append(msg)
    return True

# ════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE FRAMES
# ════════════════════════════════════════════════════════════════════════════

def extraer_frames(ruta_mp4, dir_entrenamiento, dir_calibracion,
                   intervalo_s, objetivo, log_lines):
    """
    Extrae frames del MP4 de forma intercalada entre entrenamiento y calibración.

    MODO NORMAL (frames disponibles >= 2 × objetivo):
      Frame de extracción par  → entrenamiento/
      Frame de extracción impar → calibracion/
      Resultado: exactamente `objetivo` imágenes en cada carpeta.

    MODO REPARTO EQUITATIVO (frames disponibles < 2 × objetivo):
      Ocurre cuando se ha cortado el vídeo antes de tiempo.
      En lugar de intercalar, se calcula cuántos frames hay en total y se
      distribuyen a partes iguales: los primeros N/2 → entrenamiento,
      los segundos N/2 → calibracion, con el paso ajustado para cubrir
      todo el vídeo uniformemente.
      El objetivo efectivo se reduce al máximo posible y se avisa.

    Esto garantiza que entrenamiento y calibración siempre tengan el mismo
    número de imágenes y que cubran todo el vídeo grabado, sea cual sea
    su duración.

    Los frames se guardan como JPEG con nombre que indica el prefijo
    (entr_ / cali_) y el segundo del vídeo en que fueron capturados.
    Ejemplo: entr_0012.3s.jpg = frame de entrenamiento extraído a los 12.3s del vídeo.
    Esto facilita localizar el instante original si hay que revisar una imagen.
    """
    cap = cv2.VideoCapture(ruta_mp4)
    if not cap.isOpened():
        msg = f"ERROR: No se puede abrir el vídeo: {ruta_mp4}"
        print(msg); log_lines.append(msg)
        return 0, 0

    fps_video    = cap.get(cv2.CAP_PROP_FPS) or FPS_GRABACION
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_s   = total_frames / fps_video

    msg = (f"Vídeo: {total_frames} frames  {fps_video:.1f} fps  "
           f"{duracion_s:.1f}s  ({duracion_s/60:.1f} min)")
    print(msg); log_lines.append(msg)

    paso_frames      = max(1, int(fps_video * intervalo_s))
    frames_extraibles = total_frames // paso_frames   # cuántos puntos de extracción hay

    msg = (f"Intervalo={intervalo_s}s → paso={paso_frames} frames  "
           f"extraíbles={frames_extraibles}")
    print(msg); log_lines.append(msg)

    os.makedirs(dir_entrenamiento, exist_ok=True)
    os.makedirs(dir_calibracion,   exist_ok=True)

    # ── Decidir modo de extracción ────────────────────────────────────────────
    objetivo_efectivo = objetivo

    if frames_extraibles >= objetivo * 2:
        # ── MODO NORMAL: hay de sobra, intercalar par/impar ───────────────────
        modo = "normal"
        msg = f"Modo: normal (frames suficientes para {objetivo}+{objetivo})"
        print(msg); log_lines.append(msg)

    else:
        # ── MODO REPARTO EQUITATIVO: menos frames de los necesarios ───────────
        # Calcular el máximo que cabe a partes iguales
        objetivo_efectivo = frames_extraibles // 2
        modo = "reparto"
        msg = (f"Vídeo más corto que el objetivo. "
               f"Modo reparto equitativo: {objetivo_efectivo} imágenes por carpeta "
               f"(objetivo original: {objetivo}). "
               f"Graba un vídeo más largo para obtener {objetivo}+{objetivo}.")
        print(f"⚠  {msg}"); log_lines.append(f"AVISO: {msg}")

    print(f"Extrayendo frames (modo={modo}, objetivo_por_carpeta={objetivo_efectivo})...")

    n_train = 0
    n_calib = 0

    if modo == "normal":
        # Recorrer los puntos de extracción en orden; asignar par/impar
        extraccion_idx = 0
        frame_pos      = 0

        while n_train < objetivo_efectivo or n_calib < objetivo_efectivo:
            if frame_pos >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            if extraccion_idx % 2 == 0 and n_train < objetivo_efectivo:
                seg = frame_pos / fps_video
                nombre = f"entr_{seg:08.3f}s.jpg"
                cv2.imwrite(os.path.join(dir_entrenamiento, nombre),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_CALIDAD])
                n_train += 1
                if n_train % 50 == 0:
                    print(f"  entrenamiento: {n_train}/{objetivo_efectivo}  "
                          f"calibracion: {n_calib}/{objetivo_efectivo}")

            elif extraccion_idx % 2 == 1 and n_calib < objetivo_efectivo:
                seg = frame_pos / fps_video
                nombre = f"cali_{seg:08.3f}s.jpg"
                cv2.imwrite(os.path.join(dir_calibracion, nombre),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_CALIDAD])
                n_calib += 1
                if n_calib % 50 == 0:
                    print(f"  entrenamiento: {n_train}/{objetivo_efectivo}  "
                          f"calibracion: {n_calib}/{objetivo_efectivo}")

            frame_pos      += paso_frames
            extraccion_idx += 1

    else:
        # MODO REPARTO: dividir el vídeo en dos mitades temporales del mismo tamaño.
        # Primera mitad de los frames extraíbles → entrenamiento
        # Segunda mitad → calibracion
        # Se usa un paso ajustado para distribuir los frames uniformemente
        # a lo largo de todo el vídeo en lugar de solo por la primera parte.
        #
        # Ejemplo con 360 frames extraíbles (3 min), objetivo_efectivo=180:
        #   - Los 180 puntos de entrenamiento se reparten uniformemente
        #     por todo el vídeo con paso = total_frames / 180
        #   - Los 180 puntos de calibración se intercalan entre ellos
        #   - Resultado: entrenamiento y calibración cubren todo el vídeo,
        #     sin que una cubra solo el principio y la otra solo el final.

        total_a_extraer = objetivo_efectivo * 2  # total de frames que vamos a sacar

        # Paso para distribuir total_a_extraer puntos sobre total_frames
        paso_equitativo = max(1, total_frames // total_a_extraer)

        frame_pos = 0
        idx       = 0

        while idx < total_a_extraer and frame_pos < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            if idx % 2 == 0:
                seg = frame_pos / fps_video
                nombre = f"entr_{seg:08.3f}s.jpg"
                cv2.imwrite(os.path.join(dir_entrenamiento, nombre),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_CALIDAD])
                n_train += 1
                if n_train % 50 == 0:
                    print(f"  entrenamiento: {n_train}/{objetivo_efectivo}  "
                          f"calibracion: {n_calib}/{objetivo_efectivo}")
            else:
                seg = frame_pos / fps_video
                nombre = f"cali_{seg:08.3f}s.jpg"
                cv2.imwrite(os.path.join(dir_calibracion, nombre),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_CALIDAD])
                n_calib += 1
                if n_calib % 50 == 0:
                    print(f"  entrenamiento: {n_train}/{objetivo_efectivo}  "
                          f"calibracion: {n_calib}/{objetivo_efectivo}")

            frame_pos += paso_equitativo
            idx       += 1

    cap.release()

    msg = (f"Extracción completada: entrenamiento={n_train}  calibracion={n_calib}  "
           f"(objetivo por carpeta: {objetivo_efectivo})")
    print(msg); log_lines.append(msg)

    return n_train, n_calib

# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_base = args.salida
    os.makedirs(ruta_base, exist_ok=True)

    log_lines = []
    log_lines.append(f"grabar_dataset_v1.py — sesión {ts}")
    log_lines.append(f"Directorio de salida: {ruta_base}")
    log_lines.append(f"Parámetros: num_imagenes={args.num_imagenes} "
                     f"(objetivo_por_carpeta={args.objetivo_carpeta})  "
                     f"intervalo={args.intervalo}s  duracion={args.duracion}s")

    dir_train = os.path.join(ruta_base, "entrenamiento")
    dir_calib = os.path.join(ruta_base, "calibracion")

    # ── Modo: solo extraer frames de un vídeo existente ───────────────────────
    if args.solo_extraer:
        ruta_mp4 = args.solo_extraer
        if not os.path.exists(ruta_mp4):
            print(f"ERROR: Vídeo no encontrado: {ruta_mp4}")
            sys.exit(1)
        print(f"Modo --solo-extraer: {ruta_mp4}")
        log_lines.append(f"Modo: solo extracción de {ruta_mp4}")

    # ── Modo: grabar + extraer ────────────────────────────────────────────────
    else:
        ruta_mp4      = os.path.join(ruta_base, f"giroscopio_{ts}.mp4")
        mostrar_preview = not args.sin_preview

        # ── Validación: duración mínima necesaria ──────────────────────────
        # num_imagenes × intervalo = tiempo mínimo para obtener el objetivo sin
        # activar el modo reparto equitativo.
        duracion_minima   = args.num_imagenes * args.intervalo
        frames_posibles   = int(args.duracion / args.intervalo)
        n_por_carpeta_est = frames_posibles // 2

        print(f"  Imágenes pedidas   : {args.num_imagenes} total "
              f"({args.objetivo_carpeta} train + {args.objetivo_carpeta} calib)")
        print(f"  Duración mínima    : {duracion_minima:.0f}s "
              f"({duracion_minima/60:.1f} min) para {args.num_imagenes} imágenes")

        if n_por_carpeta_est < args.objetivo_carpeta:
            print(f"\n⚠  AVISO: Con {args.duracion}s e intervalo={args.intervalo}s "
                  f"solo se obtendrán ~{n_por_carpeta_est} imágenes por carpeta "
                  f"(objetivo: {args.objetivo_carpeta}).")
            print(f"   Para {args.objetivo_carpeta} por carpeta se necesitan "
                  f"≥{duracion_minima:.0f}s ({duracion_minima/60:.1f} min).")
            print(f"   Se activará modo reparto equitativo al finalizar.")
            respuesta = input("   ¿Continuar de todas formas? [s/N]: ").strip().lower()
            if respuesta not in ("s", "si", "sí", "y", "yes"):
                print("Cancelado.")
                sys.exit(0)
        else:
            margen = args.duracion - duracion_minima
            print(f"  ✓ Duración suficiente. Margen: {margen:.0f}s — "
                  f"puedes parar con Q cuando el OSD diga OBJETIVO ALCANZADO.")

        print(f"\n{'='*60}")
        print(f"GRABACIÓN DE DATASET — giroscopio_12kp")
        print(f"{'='*60}")
        print(f"Vídeo de salida : {ruta_mp4}")
        print(f"Duración máxima : {args.duracion}s ({args.duracion/60:.1f} min)")
        print(f"Resolución      : {CAM_W}×{CAM_H}")
        print(f"Intervalo frames: {args.intervalo}s")
        print(f"Objetivo/carpeta: {args.objetivo_carpeta} imágenes  (total: {args.num_imagenes})")
        print(f"Preview         : {'SÍ (Q para detener)' if mostrar_preview else 'NO'}")
        print(f"{'='*60}\n")

        # Cuenta atrás
        print("Comenzando en 5 segundos — coloca el giroscopio en el encuadre.")
        print("Recuerda: el giroscopio debe ocupar el 40-70% del frame.")
        print("Mueve el giroscopio con variedad de roll, pitch y yaw.\n")
        for i in range(5, 0, -1):
            print(f"  {i}...")
            time.sleep(1.0)
        print()

        ok = grabar_video(ruta_mp4, args.duracion, args.intervalo,
                          args.objetivo_carpeta, log_lines, mostrar_preview)
        if not ok:
            print("ERROR en la grabación. Abortando.")
            sys.exit(1)

    # ── Extracción de frames ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EXTRACCIÓN DE FRAMES")
    print(f"{'='*60}")

    n_train, n_calib = extraer_frames(
        ruta_mp4, dir_train, dir_calib,
        args.intervalo, args.objetivo_carpeta, log_lines)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"Vídeo guardado  : {ruta_mp4}")
    print(f"Entrenamiento   : {n_train} imágenes en {dir_train}")
    print(f"Calibración     : {n_calib} imágenes en {dir_calib}")
    print()
    print("SIGUIENTES PASOS:")
    print("  1. Subir entrenamiento/ a Roboflow para etiquetar.")
    print("  2. Augmentations: solo brillo/contraste/ruido. Sin flips.")
    print("  3. Exportar en formato YOLOv8 Pose.")
    print("  4. Entrenar en Google Colab con giroscopio_colab_v4.ipynb.")
    print("  5. Usar calibracion/ como calib/ en el Docker Hailo DFC.")
    print("     OJO: las imágenes de calibracion/ son para el HEF de OTRO grupo (Bug 3).")
    print("  6. Verificar HEF: test de ruido → conv71:max>0.")
    print(f"{'='*60}\n")

    # Guardar log
    log_path = os.path.join(ruta_base, "resumen_extraccion.txt")
    log_lines.append(f"\nRESUMEN: entrenamiento={n_train}  calibracion={n_calib}")
    log_lines.append(f"Vídeo: {ruta_mp4}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Log guardado en: {log_path}")


if __name__ == "__main__":
    main()
