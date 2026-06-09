#!/usr/bin/env python3
"""
giroscopio_12kp_v3h.py — Detección y estimación de ángulos
================================================================================
Proyecto de innovación educativa — IES Politécnico Jesús Marín, Málaga
Familia Electricidad y Electrónica — Junta de Andalucía

PROPÓSITO
---------
Detectar un giroscopio físico mediante 12 keypoints (YOLOv8n-Pose compilado
para Hailo-8) y estimar en tiempo real sus ángulos de orientación:
Roll, Pitch y Yaw.

CÓMO USAR ESTE SCRIPT
----------------------
  # Con cámara (Raspberry Pi):
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef

  # Con vídeo grabado (PC o RPi sin cámara):
  python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --video /home/ai/resultado_p410.mp4

  # Guardar vídeo de salida:
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --save /home/ai/resultado_v3h_p410.mp4

  # Ocultar Yaw del panel (no fiable, recomendado para demostraciones):
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --hide-yaw

  # Si aparecen detecciones dobles, bajar el umbral IoU:
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --iou-thresh 0.25

  # Recuperar más KPs en posiciones extremas (puede añadir algo de ruido):
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --kp-thresh 0.05

  # Con ventana de delta a 5 segundos (por defecto: 2s):
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --delta-t 5.0

  # Debug extra + diagnóstico de wrap-around del delta (v3g FIX 4):
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --verbose

  # Giroscopio montado girado 180° en eje Z:
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --flip-pitch --flip-yaw

  # Combinado: demo sin Yaw, umbral IoU ajustado:
  DISPLAY=:0 python3 giroscopio_12kp_v3h.py \
    --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef \
    --hide-yaw --iou-thresh 0.25

HISTORIA DE VERSIONES Y PROBLEMAS RESUELTOS
--------------------------------------------

v1 — giroscopio_12kp_gX.py
  BUG: tensores de keypoints (conv45/59/72) pedidos como UINT8.
  El HEF los compila como UINT16. HailoRT aborta el hilo de inferencia.
  Resultado: 0 detecciones siempre.

v2 — fix UINT16
  FIX: pedir conv45/59/72 como FormatType.UINT16.
  NUEVO BUG: sigmoid aplicado a tensores de conf ya en formato PROB
  (zp=0, scale=1/255). sigmoid(0) = 0.5 → ~1500 falsas detecciones.

v3 — giroscopio_12kp_g1_0det_v3.py
  FIX: detectar tensores de conf en formato PROB y saltar sigmoid.
  NUEVO BUG: dataset de calibración del HEF = mismo dataset de entrenamiento
  → compilador calibra rango optimista → tensor de confianza devuelve raw=0
  en producción → 0 detecciones.

v4_det — giroscopio_12kp_g1_v4_det.py
  FIX: CONF_ALREADY_PROB hardcodeado con valores reales del HEF.
  FIX: CONF_THRESH bajado a 0.01 para diagnóstico.
  NUEVO: logging exhaustivo siempre activo.
  NUEVO: dump automático del tensor RAW tras 60 frames sin detección.
  NUEVO: soporte --video para pruebas sin cámara.
  RESULTADO: confirmó raw_conf = TODO CEROS → problema de compilación.
  FIX DEFINITIVO: recompilar HEF con calibración cruzada. Resuelto 16/04/2026.

v5 — giroscopio_12kp_v5.py
  Incorpora todos los fixes anteriores.
  NUEVO: panel lateral con Roll, Pitch, Yaw, confianza y FPS.
  NUEVO: --hef parametrizable.
  NUEVO: verificación automática al arranque (test de ruido).
  NUEVO: logging a fichero log_YYYYMMDD_HHMMSS.txt.

v6 — giroscopio_12kp_v6.py (== giroscopio_12kp_v3a.py — mismo código, renombrado)
  FIX KEYPOINTS: KP_SCALE reducido de 10.0 a 4.0 (valor estándar YOLOv8).
    Con scale=10 las celdas del borde proyectaban KPs ±160px fuera del frame.
    Con scale=4.0 el rango máximo es ±64px — pero resultó INSUFICIENTE para
    este modelo (el giroscopio ocupa ~600×400px de los 640×640 de entrada).
  FIX KEYPOINTS: clamp de coordenadas KP al rango [0, 640) antes de NMS.
  FIX ÁNGULOS: eliminado el -90° en PITCH.
  FIX ÁNGULOS: ROLL_OFFSET inicial = +90.0.
  FIX LOGS: eliminados mensajes DEBUG de bajo valor.
  NUEVO: carpeta historico_logs/ para logs históricos.
  PENDIENTE: KP_SCALE correcto para el nuevo modelo.

v3a — giroscopio_12kp_v3a.py (nombre confuso — mismo código que v6)
  Sin cambios funcionales respecto a v6. Solo el nombre cambió.
  Se identificaron dos problemas en las pruebas del 20/05/2026:
    1. KP_SCALE=4.0 → KPs agrupados en zona de ~65×64px en lugar de
       dispersarse por el objeto (~600×400px). Valor insuficiente.
    2. Conversión de imagen errónea antes de inferencia:
       El script hacía RGB→BGR antes de pasar al modelo (línea 777).
       El modelo fue entrenado con imágenes BGR (OpenCV estándar).
       La cámara (Picamera2) entrega RGB888. La conversión a BGR era CORRECTA.
       SIN EMBARGO: el v6 del proyecto (separado, ejecutado con el script
       propio del profe) NO hacía esa conversión y daba conf_max hasta 0.8+.
       Esto sugiere que el modelo fue entrenado con imágenes RGB o que el
       compilador Hailo absorbe la diferencia. Se añade --bgr flag para probar.

v3b — giroscopio_12kp_v3b.py (20/05/2026)
  FIX KEYPOINTS: KP_SCALE aumentado de 4.0 a 8.0.
  FIX COLOR: eliminada la conversión RGB→BGR antes de inferencia.
  NUEVO: log de coordenadas KP por cada detección en modo --verbose.
  NUEVO: log de diagnóstico al arranque mostrando KP_SCALE y rango esperado.
  FIX: nombre de ventana y logs coherentes con la versión real (v3b).
  FIX: log_dir usa directorio del script, no del proceso que lo llama.

v3c — (28/05/2026)
  NUEVO: delta de ángulos — muestra la variación de Roll/Pitch/Yaw respecto
    a hace N segundos (parametrizable con --delta-t, default 2.0 s).
    Se usa un buffer circular de (timestamp, roll, pitch, yaw). En cada frame
    se busca la entrada más antigua dentro de la ventana y se calcula la
    diferencia angular normalizada a [-180, 180].
    Motivo: el giroscopio puede estar montado rotado 180° en el eje Z respecto
    a lo definido en la estructura de KPs — los ángulos absolutos pueden ser
    engañosos dependiendo de la orientación inicial. El delta es invariante
    a la orientación inicial y permite detectar variaciones reales.
  NUEVO: colores diferenciados por eje en el panel lateral:
    PITCH → verde  (0, 220, 0)
    ROLL  → rojo   (0, 60, 220)   ← BGR
    YAW   → azul   (220, 80, 0)   ← BGR
  MEJORA: panel lateral ampliado para mostrar δRoll, δPitch, δYaw.
  NUEVO: --delta-t flag CLI para parametrizar la ventana de tiempo del delta.
  FIX CÁLCULO PITCH con giroscopio girado 180° en eje Z:
    Al girar 180° en Z, lo que era "frontal" pasa a ser "trasero" y viceversa.
    El vector mid(KP1,KP2)→mid(KP3,KP4) se invierte → Pitch saca 180° de error.
    Solución: flag --flip-pitch que invierte el signo del vector de pitch,
    equivalente a sumar/restar 180° tras normalizar. Aplica también a Yaw
    (--flip-yaw) ya que el eje de la base también queda invertido lateralmente.
    Si el dron "mira hacia atrás" con offsets a 0: usar --flip-pitch --flip-yaw.
  BUGS CONOCIDOS (observados en pruebas p211 y p113, 01/06/2026):
    1. Símbolos "?" en el panel — \u00b0, \u2191, \u2193, \u2194, \u0394
       no son renderizables por cv2.putText con FONT_HERSHEY (solo ASCII).
    2. Roll incorrecto (~-180° en reposo). El eje correcto existe en el modelo
       pero los offsets y la lógica de fallback necesitan revisión.
    3. Pitch funciona bien solo en ±90°. En posiciones intermedias la
       perspectiva lateral de la cámara acorta el vector y añade ruido.
    4. Yaw confirmado como no fiable con una sola cámara lateral.

v3d — (01/06/2026)
  FIX CRÍTICO — Símbolos no ASCII en panel (Bug 13): sustituidos por ASCII puro.
  FIX ROLL — lógica cambiada a vectores por lado KP5→KP7 / KP6→KP8 con promedio
    circular (más robusto que mid(KP5,KP6)→mid(KP7,KP8)).
  FIX PITCH — suavizado EMA configurable (ANGLE_SMOOTH_ALPHA=0.3).
  MEJORA YAW — marcado como "YAW(N/D)" en panel (no fiable con cámara lateral única).
  NUEVO: log de diagnóstico de roll en modo --verbose (Y de KP5-8).
  BUG CONOCIDO (Bug 16): EMA puede producir valores fuera de [-180, 180] en wrap-around.
  BUG CONOCIDO (Bug 14/15): ROLL_OFFSET y PITCH_OFFSET no calibrados para este modelo.

v3e — (01/06/2026)
  FIX Bug 16 — re-normalización EMA a [-180, 180].
  FIX Bug 14 — ROLL_OFFSET = 175.0 (intento de calibración, resultó incorrecto).
  FIX Bug 15 — PITCH_OFFSET = -90.0 (correcto — verificado en pruebas v3e).
  NUEVO: WINDOW_SCALE_FACTOR=2 restaurado.
  BUG RESIDUAL: ROLL_OFFSET=175 da -90deg en reposo en lugar de 0deg.
    Diagnóstico post-pruebas: el raw del roll es ~+95deg (no -90deg como se asumía).
    Eso significa que el vector KP5→KP7 apunta hacia ARRIBA en imagen (Y_KP5 > Y_KP7),
    es decir, el modelo tiene los KPs del soporte verticalmente invertidos, igual que los
    del octógono. Causa: misma perspectiva de grabación por detrás (Bug 17 a continuación).

v3f — (01/06/2026)
  FIX Bug 17 — ROLL_OFFSET corregido de 175.0 a -95.0:
    ANÁLISIS COMPLETO (derivado de pruebas v3e, 01/06/2026):
    - Con v3d (ROLL_OFFSET=+90):  pantalla = -175deg en reposo → raw ≈ +95deg
    - Con v3e (ROLL_OFFSET=+175): pantalla = -90deg en reposo  → raw ≈ +95deg
    - Conclusión: el raw del modelo es consistentemente ~+95deg (no -90deg).
    - Por qué raw=+95 y no -90: el vector KP5→KP7 apunta hacia ARRIBA en imagen
      (KP5 tiene Y mayor que KP7 en píxeles), dando arctan2(+dy, dx) ≈ +90°.
      Esto ocurre porque el modelo detecta KP5 ("arriba") en posición más baja en
      imagen que KP7 ("abajo") — misma inversión vertical que los KPs del octógono,
      causada por el vídeo de entrenamiento grabado por la parte trasera del giroscopio.
    - Offset correcto: -95° → pantalla = fix_angle(+95, -95) = 0deg ✓
      Redondeado a -90° para alinearse con la geometría real (arctan2 puro = ±90°).
    ATENCIÓN: este offset -90 es específico del modelo actual (211 imágenes, perspectiva
    invertida). Con el nuevo dataset grabado de frente, el vector apuntará hacia abajo
    (raw = -90deg) y el offset correcto volverá a ser +90 (el valor geométrico "natural").
    Los comentarios en el código documentan ambos casos.
  CONFIRMADO: PITCH_OFFSET = -90.0 es correcto (pruebas v3e muestran +2 a +3deg en reposo).
  SIN CAMBIOS en lógica de cálculo, EMA, panel ni ninguna otra parte.

v3g — (03/06/2026)
v3h — (03/06/2026)
v3h — ESTE SCRIPT (03/06/2026, rev.2 — optimización FPS)
  FIX RENDIMIENTO: cv2.addWeighted sobre ROI en lugar de frame completo.

  PROBLEMA: la versión anterior hacía frame.copy() completo (960×540×3 px)
  en cada llamada a _badge() y en el fondo del panel. Con 3 badges + 1 fondo
  = 4 copias de ~1.5M píxeles por frame → caída de FPS de ~24 a ~15.

  CAUSA TÉCNICA: cv2.addWeighted necesita dos arrays del mismo tamaño. La
  implementación naive copia el frame entero para tener un array de trabajo.

  SOLUCIÓN: usar vistas NumPy (ROI = frame[y0:y1, x0:x1]).
  frame[y0:y1, x0:x1] devuelve una vista del array original, no una copia.
  addWeighted sobre la vista escribe directamente en esa región del frame
  sin copiar nada. Coste: solo los píxeles del ROI.

  Ahorro por frame:
    - Panel (310×340px):    frame.copy() 1.5M px → roi.copy() ~316K px  (−80%)
    - Cada badge (~100×30): frame.copy() 1.5M px → roi.copy() ~9K px    (−99%)
    - 3 badges + panel: 6M px → ~343K px → reducción ~94%

  SIN CAMBIOS en: lógica de cálculo, pipeline de inferencia, visualización.


  NUEVO: panel de actitud visual con arcos OpenCV.

  MOTIVACIÓN: los números absolutos de ángulo (ej. +2.9deg) tienen un ruido
  inherente de ±5-7deg medido en pruebas físicas (stdev Roll=6.98°, Pitch=3.26°
  con el giroscopio en reposo). Mostrarlos con decimales da una falsa sensación
  de precisión. Lo que el sistema SÍ puede dar con fiabilidad es:
    - ¿Cabecea hacia adelante o atrás?    → Pitch
    - ¿Alabea a derecha o izquierda?       → Roll
    - ¿Gira en sentido horario o antihorario? → Yaw delta

  CAMBIOS respecto a v3g:
  - draw_results() reemplazada completamente por draw_results_visual().
  - Panel lateral reemplazado por tres indicadores visuales con arcos/péndulo:
      · PITCH: arco semicircular superior. Punto coloreado se mueve a lo largo
        del arco: centro=nivel, derecha=adelante, izquierda=atrás.
      · ROLL: péndulo. Aguja gira alrededor del eje central. Visualmente
        intuitivo como un nivel de burbuja.
      · YAW: flecha en círculo. Indica dirección del delta (no ángulo absoluto).
        Siempre etiquetado como (N/D) — no fiable con cámara lateral única.
  - Badge de texto bajo cada arco: "NIVEL", "ADELANTE", "ATRAS", "DER", "IZQ",
    "HORARIO", "ANTIHOR" con fondo de color (verde/rojo/gris).
  - Números siguen visibles, pequeños (fs=0.38), bajo el badge.
  - Zona muerta configurable: DEAD_ZONE_DEG (default 6°). Dentro de la zona
    muerta el indicador muestra color gris neutro y badge "NIVEL"/"QUIETO"/N/D.
    Este valor absorbe el ruido de ±5-7deg observado en pruebas físicas.
  - Cabecera (Det, Conf, FPS) se mantiene en la esquina superior izquierda
    del panel, compacta, tamaño reducido.
  - NUEVO: --dead-zone flag CLI para ajustar la zona muerta en grados.
  - SIN CAMBIOS en pipeline de inferencia, keypoints, EMA, delta, logs.


  AJUSTES POST-PRUEBAS FÍSICAS p309/p410 (análisis de capturas y logs del 03/06/2026):
  Las pruebas confirmaron p410 como mejor modelo: confianza media 0.888 (vs 0.636 de p309),
  0 frames sin detección (vs 30 consecutivos en p309), 99.5% frames >0.5 conf.
  Se identificaron 6 problemas y se resuelven los 4 que son de código:

  FIX 1 — Detecciones dobles: IOU_THRESH bajado de 0.45 a 0.30.
    CAUSA: la estructura metálica del giroscopio genera dos bounding boxes con
    solapamiento <45% en ciertos ángulos. Con 0.30 se fusionan correctamente.
    Observado en pruebas p410 frame 34 (Det: 2, Conf: 0.902 + 0.047).
    NUEVO: --iou-thresh parametrizable en CLI para ajuste fino.

  FIX 2 — Falsas detecciones débiles: CONF_THRESH subido de 0.01 a 0.05.
    CAUSA: con conf=0.01 se aceptaban detecciones de ruido (0.02-0.03) que nunca
    corresponden al giroscopio real. p410 tiene conf_media=0.888, así que 0.05
    no pierde ninguna detección legítima. p309 con conf_media=0.636 tampoco pierde.
    Diagnóstico: el valor 0.01 era necesario en las fases de debugging (v4_det),
    ya no tiene sentido en producción.

  FIX 3 — KPs perdidos en perspectiva extrema: KP_THRESH bajado de 0.10 a 0.07.
    CAUSA: los KPs traseros del octógono (KP1 blanco, KP2 negro) tienen visibilidad
    0.07-0.09 cuando están parcialmente ocultos por perspectiva lateral. Con 0.10 se
    pierden y el cálculo de Pitch cae al fallback de 2 puntos (menos preciso).
    Con 0.07 se recuperan sin añadir ruido significativo.

  FIX 4 — Log de delta raw en modo verbose.
    NUEVO: en modo --verbose, cada frame logea los valores brutos de delta_angle()
    para diagnosticar posibles problemas de wrap-around (Bug 16 residual).
    Formato: [DELTA-DIAG] d_roll=X d_pitch=X d_yaw=X

  NUEVO: --iou-thresh parametrizable en CLI (default: 0.30).
  NUEVO: --hide-yaw flag para ocultar la fila de Yaw del panel.
    Motivo: el Yaw no es fiable con una sola cámara lateral y su presencia
    confunde en demostraciones. Con --hide-yaw se oculta completamente.
    Sin el flag, se sigue mostrando como YAW(N/D) para diagnóstico.
  NUEVO: --kp-thresh parametrizable en CLI (default: 0.07).

  SIN CAMBIOS en: lógica de cálculo de ángulos, EMA, pipeline de inferencia,
  colores de KPs, estructura de tensores, verificación del HEF.

  PROBLEMAS NO RESUELTOS (requieren hardware o reentrenamiento):
  - Yaw no fiable con cámara lateral única → requiere 2ª cámara o IMU.
  - stride 8 (conv44) siempre raw_conf=0 → recalibrar HEF con objetos pequeños.
  - Inestabilidad KP1/KP2 en Pitch >70° → limitación geométrica de perspectiva.

BUG DE CALIBRACIÓN DEL HEF — DOCUMENTADO
-----------------------------------------
SÍNTOMA: raw_conf = 0 en todos los frames para todos los strides.
         bbox y kps tienen valores normales. 0 detecciones siempre.
CAUSA: dataset de calibración = mismo dataset de entrenamiento.
       El compilador calibra el rango de la capa de confianza con
       imágenes "conocidas" → rango demasiado optimista → en producción
       todos los valores caen en raw=0.
SOLUCIÓN: recompilar con imágenes de OTRAS escenas como calibración.
       El nuevo modelo (220 img entrenamiento + 560 con augmentations)
       funcionó correctamente con calib2 y calib3.
VERIFICACIÓN: test de ruido al arranque → conv71:max debe ser > 0.
       calib1: max=7  ✓
       calib2: max=24 ✓
       calib3: max=11 ✓
       estandar (mismo ds que train): max=0 ✗ → no usar

ESTRUCTURA DE 12 KEYPOINTS
---------------------------
KP#  Grupo  Color          Posición                  Eje/Ángulo
1    OCT    blanco         Esq trasera-izq octógono  PITCH
2    OCT    negro          Esq trasera-der octógono  PITCH
3    OCT    rojo           Esq frontal-der octógono  PITCH
4    OCT    amarillo       Esq frontal-izq octógono  PITCH
5    INT    azul eléctrico Punta U arriba-izq         ROLL
6    INT    naranja        Travesaño arriba-der       ROLL
7    INT    verde lima     Punta U abajo-izq          ROLL
8    INT    magenta        Travesaño abajo-der        ROLL
9    BASE   cian           Esq trasera-izq base       YAW
10   BASE   rosa           Esq trasera-der base       YAW
11   BASE   marrón         Esq frontal-der base       YAW
12   BASE   morado         Esq frontal-izq base       YAW

CÁLCULO DE ÁNGULOS
------------------
PITCH (KP1-4): vector mid(KP1,KP2) → mid(KP3,KP4). Sin restar 90°.
ROLL  (KP5-8): vector mid(KP5,KP6) → mid(KP7,KP8). ROLL_OFFSET=+90° compensa arctan2.
YAW   (KP9-12): vector mid(KP9,KP12) → mid(KP10,KP11).

TENSORES HEF (giroscopio.hef — nuevo modelo 20/05/2026)
--------------------------------------------------------
conv43/57/70  UINT8  FCR(H×W×64)   bbox DFL  (strides 8, 16, 32)
conv44/58/71  UINT8  NHWC(H×W×1)   confianza — formato PROB (zp=0, scale=1/255)
conv45/59/72  UINT16 FCR(H×W×36)   keypoints (12KP × 3 = 36 valores)

Tensores de conf en formato PROB (zp=0): raw=0→0.0, raw=128→0.5, raw=255→1.0.
NO aplicar sigmoid.
"""

import argparse
import datetime
import logging
import os
import re
import sys
import threading
import time
import traceback
import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    HAVE_PICAMERA = True
except ImportError:
    HAVE_PICAMERA = False

try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                                 ConfigureParams, InputVStreamParams,
                                 OutputVStreamParams, FormatType, InferVStreams)
    HAVE_HAILO = True
except ImportError:
    HAVE_HAILO = False

os.environ.setdefault("DISPLAY", ":0")
os.environ.pop("WAYLAND_DISPLAY", None)

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — todos los parámetros relevantes aquí arriba
# ════════════════════════════════════════════════════════════════════════════

VERSION = "v3h"

# ── Arquitectura del modelo ───────────────────────────────────────────────────
PREFIX  = "giroscopio/"   # prefijo de tensores en el HEF (nombre de red)
NUM_KP  = 12              # número de keypoints del modelo

# ── Cámara ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 960, 540  # resolución de captura Picamera2
MODEL_SIZE   = 640        # tamaño de entrada del modelo (cuadrado)

# ── Umbrales de detección ─────────────────────────────────────────────────────
# Con los HEF del nuevo modelo (220 img + calib cruzada) la conf es baja:
#   calib2/3 con v3c: esperamos conf_max en rango 0.012–0.027 inicialmente
#   el script profe (v6 sin conversión BGR) llegó hasta 0.7–0.8 con calib1
# Mantener CONF_THRESH bajo para no perder detecciones débiles.
CONF_THRESH = 0.05    # umbral objectness — subido de 0.01 en v3g (ver FIX 2)
KP_THRESH   = 0.07    # umbral visibilidad KP — bajado de 0.10 en v3g (ver FIX 3)
IOU_THRESH  = 0.30    # NMS: IoU máximo — bajado de 0.45 en v3g (ver FIX 1)

# ── Escala de keypoints ───────────────────────────────────────────────────────
# HISTORIA:
#   10.0 → rango máximo ±160px con stride=32. KPs salían del frame en celdas
#          de borde. Problemático con el HEF mal calibrado (v5).
#   4.0  → rango máximo ±64px. Insuficiente para objeto de ~600px en pantalla.
#          KPs se agrupaban en zona de ~65×64px (observado 17/04/2026 y 20/05/2026).
#   8.0  → rango máximo ±128px. Equilibrio razonable. A verificar.
#
# CÓMO DIAGNOSTICAR:
#   Usar --verbose y buscar en el log la línea "[KP-RANGE]".
#   Si rango_X ≈ rango_Y ≈ 65px → subir KP_SCALE.
#   Si los KPs aparecen fuera del bounding box → bajar KP_SCALE.
KP_SCALE = 8.0

# ── Color de imagen al modelo ─────────────────────────────────────────────────
# El nuevo modelo (entrenado en Roboflow/Colab con PIL/albumentations) recibe
# imágenes en formato RGB. Usar False. Si los resultados son malos, probar True.
# En CLI: --bgr para forzar conversión RGB→BGR.
CONVERT_TO_BGR = False  # False = pasar RGB tal cual; True = convertir a BGR

# ── Offsets de ángulos en reposo ──────────────────────────────────────────────
# PENDIENTE DE CALIBRAR con el nuevo modelo.
# Procedimiento: poner a 0, ejecutar con giroscopio en reposo y nivelado,
# anotar valores del panel, poner offset = -(valor medido).
ROLL_OFFSET  = -90.0
# ^ v3f: corregido tras análisis de pruebas v3e (01/06/2026).
#
#   HISTORIA DEL OFFSET DE ROLL:
#   - v3b/v3c/v3d: ROLL_OFFSET=+90.0 → pantalla=-175deg en reposo
#   - v3e:         ROLL_OFFSET=+175.0 → pantalla=-90deg en reposo
#   - v3f:         ROLL_OFFSET=-90.0  → pantalla=~0deg en reposo  ← CORRECTO
#
#   ANÁLISIS: el raw del modelo para este dataset es ~+95deg (no -90deg).
#   El vector KP5→KP7 apunta hacia ARRIBA en imagen (KP5 tiene Y > KP7 en píxeles)
#   porque el modelo aprendió los KPs del soporte con inversión vertical — misma causa
#   que los KPs del octógono: el vídeo de entrenamiento se grabó por detrás del giroscopio.
#   arctan2(-(KP7.y - KP5.y), ...) con KP7.y < KP5.y → argumento positivo → +90°
#   fix_angle(+90, -90) = 0° ✓
#
#   CON EL NUEVO DATASET (grabado de frente):
#   El vector KP5→KP7 apuntará hacia ABAJO (KP5 arriba, KP7 abajo, Y_KP5 < Y_KP7)
#   → raw = -90deg → offset correcto = +90.0 (el valor "natural" geométrico)
#   ⚠ RECALIBRAR: poner ROLL_OFFSET=0, ejecutar en reposo, anotar valor, offset=-(valor)

PITCH_OFFSET = -90.0
# ^ v3e/v3f: CONFIRMADO correcto (pruebas v3e, 01/06/2026).
#   En reposo muestra +2 a +3deg → dentro del margen de calibración aceptable.
#   Causa geométrica: cámara lateral + perspectiva invertida (grabación por detrás)
#   → el vector mid(KP1,KP2)→mid(KP3,KP4) da ~+90deg en reposo → offset=-90 → 0deg.
#   ⚠ RECALIBRAR con el nuevo modelo: poner 0.0, medir en reposo, offset=-(valor).

YAW_OFFSET   = 0.0
# ^ Yaw marcado N/D en el panel — no calibrar hasta tener segunda cámara.

# ── Escala de ventana OpenCV ──────────────────────────────────────────────────
# ── Zona muerta para los indicadores visuales (v3h) ──────────────────────────
# Ángulos dentro de este rango se muestran como "quieto/nivel/N/D" en los badges.
# El ruido de ±5-7° medido en pruebas físicas (03/06/2026) justifica un valor
# mínimo de 6°. Subir si el indicador parpadea demasiado en reposo.
DEAD_ZONE_DEG = 6.0  # grados — ajustable con --dead-zone en CLI

# v3e: restaurado desde v3b (desapareció en v3c/v3d).
# 1 = tamaño nativo de la fuente (960×540).
# 2 = 1920×1080 — recomendado para monitores FullHD.
# 3 = 2880×1620 — para pantallas 4K (valor original de v3b, demasiado grande en FullHD).
WINDOW_SCALE_FACTOR = 2

# ── Suavizado temporal de ángulos (EMA — media exponencial móvil) ─────────────
# Alpha=1.0 → sin suavizado (valor instantáneo, comportamiento v3c).
# Alpha=0.3 → suavizado moderado: reduce jitter sin añadir lag excesivo.
# Alpha=0.1 → suavizado fuerte: útil si los KPs son muy ruidosos.
# Sólo se aplica a Roll y Pitch (el Yaw no es fiable con esta perspectiva).
ANGLE_SMOOTH_ALPHA = 0.3   # factor EMA: 0 < alpha <= 1.0

# ── Inversión de ejes (giroscopio montado girado 180° en eje Z) ───────────────
# Contexto (28/05/2026): el vídeo de entrenamiento se grabó con el giroscopio
# girado 180° en Z respecto a la convención de KPs (KP1/2 que deberían ser
# "traseros" quedaron delante). Esto invierte los vectores de PITCH y YAW.
# Síntoma típico: con el dron en reposo y offsets a 0, Pitch y Yaw dan ~180°
# de error, o el dron aparece "mirando hacia atrás".
# Solución: activar --flip-pitch y --flip-yaw para invertir el vector antes
# del cálculo de arctan2 (equivalente a reflejar el vector, no a sumar 180°).
FLIP_PITCH = False  # True = invertir sentido del vector de pitch
FLIP_YAW   = False  # True = invertir sentido del vector de yaw

# ── Delta de ángulos ──────────────────────────────────────────────────────────
# El panel muestra δPitch, δRoll, δYaw = ángulo_actual - ángulo_hace_DELTA_T_S.
# Es invariante a la orientación inicial del giroscopio — útil cuando el montaje
# no sigue exactamente la convención de KPs o hay ambigüedad de 180°.
# Parametrizable con --delta-t en CLI.
# El buffer circular almacena (timestamp, roll, pitch, yaw). En cada frame se
# busca la entrada más antigua dentro de la ventana DELTA_T_S y se hace la
# diferencia angular normalizada a [-180, 180] (función delta_angle).
DELTA_T_S      = 2.0    # ventana temporal del delta en segundos (default: 2s)
DELTA_BUFFER_S = 10.0   # historial máximo en buffer (siempre > DELTA_T_S)

# ── Colores BGR del panel lateral (v3c: diferenciados por eje) ────────────────
# PITCH → verde  | ROLL → rojo  | YAW → azul
# Nota: OpenCV usa BGR, no RGB.
COLOR_PITCH = (  0, 220,   0)   # verde
COLOR_ROLL  = (  0,  60, 220)   # rojo
COLOR_YAW   = (220,  80,   0)   # azul

# ── Tensores de conf en formato PROB (NO aplicar sigmoid) ─────────────────────
# Confirmado en todos los HEF probados: zp=0, scale=1/255.
# Se inicializa vacío y se rellena automáticamente al leer el HEF.
CONF_ALREADY_PROB = set()

# ── Parámetros de dequantización (fallback si quant_info no accesible) ────────
# Se actualizan automáticamente desde el HEF. Estos son valores de referencia
# del HEF calib2 del 20/05/2026 (log 093942).
QUANT = {
    "conv43": (0.165420, 117.0),
    "conv44": (0.003922,   0.0),
    "conv45": (0.001043, 14244.0),
    "conv57": (0.105872, 120.0),
    "conv58": (0.003922,   0.0),
    "conv59": (0.000634, 15453.0),
    "conv70": (0.090270, 133.0),
    "conv71": (0.003922,   0.0),
    "conv72": (0.000564, 16619.0),
}

# ── Colores BGR de los 12 keypoints ──────────────────────────────────────────
# Orden: KP1..KP12 según la tabla de estructura de arriba.
KP_COLORS = [
    (255, 255, 255),  # KP1  blanco  — OCT trasera-izq   (PITCH)
    ( 30,  30,  30),  # KP2  negro   — OCT trasera-der   (PITCH)
    ( 16,  16, 232),  # KP3  rojo    — OCT frontal-der   (PITCH)
    (  0, 214, 255),  # KP4  amarillo — OCT frontal-izq  (PITCH)
    (255,  85,   0),  # KP5  azul el.— INT U arriba-izq  (ROLL)
    (  0, 106, 255),  # KP6  naranja — INT travesaño arr (ROLL)
    (  0, 192,  57),  # KP7  v.lima  — INT U abajo-izq   (ROLL)
    (204,   0, 204),  # KP8  magenta — INT travesaño abj (ROLL)
    (204, 187,   0),  # KP9  cian    — BASE trasera-izq  (YAW)
    (153,  68, 255),  # KP10 rosa    — BASE trasera-der  (YAW)
    (  0,  68, 136),  # KP11 marrón  — BASE frontal-der  (YAW)
    (204,   0, 102),  # KP12 morado  — BASE frontal-izq  (YAW)
]

# ════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════════════════

def setup_logging(hef_path="", log_dir_override=""):
    """
    Configura logging dual: fichero (INFO) + consola (WARNING).

    NOMENCLATURA DEL FICHERO (rev. 08/06/2026):
        logs_giroscopio_v3a_<modelo>_<YYYYMMDD_HHMMSS>.txt
        Ejemplo: logs_giroscopio_v3a_p410_20260608_143022.txt

    El <modelo> se extrae del directorio padre del HEF:
        /home/ai/giroscopio/giroscopio_v3a/p410/giroscopio.hef
                                             ^^^^ -> "p410"

    DIRECTORIO DE LOGS:
        - Si se pasa log_dir_override, se usa ese directorio.
        - Si no, se guarda en <dir_del_script>/historico_logs/
          (comportamiento original, compatibilidad con ejecucion directa).

    CAMBIO vs version anterior:
        Antes: log_gX_YYYYMMDD_HHMMSS.txt  (patron _gN del nombre HEF)
        Ahora: logs_giroscopio_v3a_<modelo>_YYYYMMDD_HHMMSS.txt
               con modelo extraido del directorio padre del HEF
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extraer nombre del modelo del directorio padre del HEF
    # .../giroscopio_v3a/p410/giroscopio.hef -> "p410"
    hef_abs = os.path.abspath(hef_path) if hef_path else ""
    modelo  = os.path.basename(os.path.dirname(hef_abs)) if hef_abs else "modelo"
    if not modelo or modelo in (".", ""):
        modelo = "modelo"

    # Directorio de logs: externo (desde wrapper .sh) o por defecto junto al script
    if log_dir_override:
        log_dir = log_dir_override
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir  = os.path.join(base_dir, "historico_logs")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"logs_giroscopio_v3a_{modelo}_{ts}.txt")

    log = logging.getLogger("gyro")
    log.setLevel(logging.DEBUG)

    # Fichero: INFO y superior (sin DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"))

    # Consola: solo WARNING/ERROR (sin ruido en producción)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    log.addHandler(fh)
    log.addHandler(ch)
    log.info(f"Log → {log_path}")
    return log, log_path

# ════════════════════════════════════════════════════════════════════════════
#  ARGUMENTOS
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=f"Giroscopio 12KP {VERSION} — Hailo-8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  Cámara:      DISPLAY=:0 python3 %(prog)s --hef /home/ai/giroscopio_v3a/p410/giroscopio.hef
  Vídeo:       python3 %(prog)s --hef giroscopio.hef --video resultado_p410.mp4
  Guardar:     DISPLAY=:0 python3 %(prog)s --hef giroscopio.hef --save resultado_v3h.mp4
  Sin Yaw:     DISPLAY=:0 python3 %(prog)s --hef giroscopio.hef --hide-yaw
  IoU ajust.:  DISPLAY=:0 python3 %(prog)s --hef giroscopio.hef --iou-thresh 0.25
  KP sensible: DISPLAY=:0 python3 %(prog)s --hef giroscopio.hef --kp-thresh 0.05
  Debug delta: DISPLAY=:0 python3 %(prog)s --hef giroscopio.hef --verbose
  Delta 5s:    DISPLAY=:0 python3 %(prog)s --hef giroscopio.hef --delta-t 5.0
  Girado 180°: DISPLAY=:0 python3 %(prog)s --hef giroscopio.hef --flip-pitch --flip-yaw
  Demo limpia: DISPLAY=:0 python3 %(prog)s --hef giroscopio.hef --hide-yaw --iou-thresh 0.25
        """)
    p.add_argument("--hef",     required=True,
                   help="Ruta al archivo HEF compilado")
    p.add_argument("--video",   default="",
                   help="Vídeo de entrada en lugar de cámara")
    p.add_argument("--save",    default="",
                   help="Ruta para guardar vídeo de salida")
    p.add_argument("--conf",    type=float, default=CONF_THRESH,
                   help=f"Umbral de confianza (default: {CONF_THRESH})")
    p.add_argument("--iou-thresh", type=float, default=IOU_THRESH,
                   help=f"Umbral IoU para NMS (default: {IOU_THRESH}). "
                        f"Bajar si hay detecciones dobles.")
    p.add_argument("--kp-thresh", type=float, default=KP_THRESH,
                   help=f"Umbral visibilidad KP individual (default: {KP_THRESH}). "
                        f"Bajar para recuperar KPs en posiciones extremas.")
    p.add_argument("--hide-yaw", action="store_true",
                   help="Ocultar Yaw del panel (no fiable con cámara lateral única)")
    p.add_argument("--dead-zone", type=float, default=DEAD_ZONE_DEG,
                   help=f"Zona muerta en grados para los indicadores (default: {DEAD_ZONE_DEG}). "
                        f"Dentro de este rango el badge muestra 'quieto/nivel'. "
                        f"Subir si parpadea en reposo, bajar si no reacciona.")
    p.add_argument("--kp-scale", type=float, default=KP_SCALE,
                   help=f"Factor escala keypoints (default: {KP_SCALE}). "
                        f"Subir si KPs agrupados, bajar si salen del objeto.")
    p.add_argument("--bgr",     action="store_true",
                   help="Convertir RGB→BGR antes de inferencia (probar si los KPs fallan)")
    p.add_argument("--flip-pitch", action="store_true",
                   help="Invertir sentido del vector de Pitch (usar si giroscopio girado 180° en Z)")
    p.add_argument("--flip-yaw",   action="store_true",
                   help="Invertir sentido del vector de Yaw (usar junto a --flip-pitch si girado 180° en Z)")
    p.add_argument("--delta-t", type=float, default=DELTA_T_S,
                   help=f"Ventana temporal del delta de ángulos en segundos (default: {DELTA_T_S})")
    p.add_argument("--verbose", action="store_true",
                   help="Log detallado por frame (incluye rango de coordenadas KP)")
    p.add_argument("--log-dir", default="",
                   help="Directorio donde guardar los logs. "
                        "Si no se especifica, se usa <dir_script>/historico_logs/. "
                        "Normalmente pasado por el wrapper .sh.")
    p.add_argument("--save-auto", default="",
                   help="Directorio donde guardar el video de salida con nombre automatico. "
                        "Nombre: giroscopio_v3a_<modelo>_<YYYYMMDD_HHMMSS>.mp4. "
                        "Incompatible con --save (que usa ruta completa manual).")
    return p.parse_args()

# ════════════════════════════════════════════════════════════════════════════
#  UTILIDADES NUMÉRICAS
# ════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))

def softmax_dfl(x, reg_max=16):
    """DFL: Distribution Focal Loss. Decodifica bbox como combinación convexa de distancias."""
    x  = x.reshape(*x.shape[:-1], 4, reg_max)
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    sm = ex / ex.sum(axis=-1, keepdims=True)
    return (sm * np.arange(reg_max, dtype=np.float32)).sum(axis=-1)

def dequant(tensor_raw, name, log):
    """Dequantización lineal: (raw - zp) * scale."""
    if name not in QUANT:
        log.error(f"dequant: '{name}' no está en QUANT")
        raise KeyError(name)
    s, zp = QUANT[name]
    return (tensor_raw.astype(np.float32) - zp) * s

def nms(boxes, scores, iou_thresh=IOU_THRESH):
    """Non-Maximum Suppression estándar por IoU."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        inter = (np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])) *
                 np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])))
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep

# ════════════════════════════════════════════════════════════════════════════
#  DECODIFICACIÓN
# ════════════════════════════════════════════════════════════════════════════

def decode_stride(raw_bbox, raw_conf, raw_kps, stride,
                  b_nm, c_nm, k_nm, conf_thresh, kp_scale, log, fid=0, verbose=False):
    """
    Decodifica un nivel de stride del modelo YOLOv8-Pose.
    Devuelve (boxes_xyxy, confs, keypoints_xy_score, conf_max).

    Fórmula de keypoints:
      kp_x = (sigmoid(raw_x) * kp_scale - kp_scale/2 + cell_x) * stride
      kp_y = (sigmoid(raw_y) * kp_scale - kp_scale/2 + cell_y) * stride

    Con kp_scale=8.0 y stride=32:
      rango de desplazamiento = ±4 celdas * 32px = ±128px desde el centro de celda.
    Con kp_scale=8.0 y stride=8:
      rango = ±4 celdas * 8px = ±32px desde el centro de celda.
    """
    H, W = raw_conf.shape[:2]

    bbox_f = dequant(raw_bbox, b_nm, log)
    conf_f = dequant(raw_conf, c_nm, log)
    kps_f  = dequant(raw_kps,  k_nm, log)

    # Mapa de confianza — NO aplicar sigmoid si ya es formato PROB (zp=0)
    raw_vals = conf_f[..., 0]
    if c_nm in CONF_ALREADY_PROB:
        conf_map = np.clip(raw_vals, 0.0, 1.0)
    else:
        conf_map = sigmoid(raw_vals)

    conf_max = float(conf_map.max())

    mask = conf_map > conf_thresh
    if not mask.any():
        return (np.empty((0, 4), np.float32),
                np.empty(0, np.float32),
                np.empty((0, NUM_KP, 3), np.float32),
                conf_max)

    ys, xs = np.where(mask)
    confs   = conf_map[ys, xs]

    # BBox via DFL → coordenadas absolutas en espacio 640×640
    ltrb  = softmax_dfl(bbox_f[ys, xs]) * stride
    cx, cy = (xs + 0.5) * stride, (ys + 0.5) * stride
    boxes  = np.stack([cx - ltrb[:,0], cy - ltrb[:,1],
                       cx + ltrb[:,2], cy + ltrb[:,3]], axis=1)

    # Keypoints
    half    = kp_scale / 2.0
    kps_sel = kps_f[ys, xs].reshape(-1, NUM_KP, 3)
    kp_x    = (sigmoid(kps_sel[..., 0]) * kp_scale - half + xs[:, None]) * stride
    kp_y    = (sigmoid(kps_sel[..., 1]) * kp_scale - half + ys[:, None]) * stride
    kp_s    = sigmoid(kps_sel[..., 2])

    # Clamp al rango válido del frame
    kp_x = np.clip(kp_x, 0.0, float(MODEL_SIZE - 1))
    kp_y = np.clip(kp_y, 0.0, float(MODEL_SIZE - 1))

    keypoints = np.concatenate([kp_x[..., None], kp_y[..., None], kp_s[..., None]], axis=-1)

    if verbose and len(kp_x) > 0:
        # Log de rango de coordenadas para diagnóstico de KP_SCALE
        log.debug(f"[KP-RANGE s={stride} f={fid}] "
                  f"kp_x=[{kp_x.min():.1f}, {kp_x.max():.1f}] "
                  f"kp_y=[{kp_y.min():.1f}, {kp_y.max():.1f}] "
                  f"rango_X={kp_x.max()-kp_x.min():.1f} "
                  f"rango_Y={kp_y.max()-kp_y.min():.1f}")

    return boxes, confs, keypoints, conf_max

# ════════════════════════════════════════════════════════════════════════════
#  CÁLCULO DE ÁNGULOS
# ════════════════════════════════════════════════════════════════════════════

def fix_angle(v, offset):
    """Aplica offset y normaliza al rango [-180, 180]."""
    if v is None: return None
    return ((v + offset + 180) % 360) - 180

def calc_angles(kps, scores, flip_pitch=False, flip_yaw=False, verbose_log=None, kp_thresh=KP_THRESH):
    """
    Calcula Roll, Pitch, Yaw a partir de los 12 keypoints visibles.
    kps: array (12, 2) — coordenadas X,Y en espacio 640×640
    scores: array (12,) — visibilidad de cada KP
    flip_pitch: invertir sentido del vector de pitch (para giroscopio girado 180° en Z)
    flip_yaw:   invertir sentido del vector de yaw   (ídem)
    verbose_log: logger para diagnóstico del vector de roll (None = silencio)

    CAMBIOS v3d respecto a v3c:
      - ROLL: cambiado de vector mid(KP5,KP6)→mid(KP7,KP8) a vectores por lado:
        KP5→KP7 (lado izquierdo, punta-U) y KP6→KP8 (lado derecho, travesaño).
        El ángulo es el promedio de ambos vectores (más robusto). Si solo hay
        un lado disponible se usa ese. El vector vertical en reposo apunta
        hacia abajo en imagen → arctan2 = -90° → ROLL_OFFSET=+90° da 0°.
        Diagnóstico: en modo verbose se loguea la Y de cada par para detectar
        si el modelo invierte "arriba" y "abajo".
      - Roll/Pitch ahora devuelven el valor RAW (sin EMA); el suavizado se
        aplica en el bucle de inferencia con _smooth_angle().
    """
    def vis(i):      return float(scores[i]) > kp_thresh
    def mid(i, j):   return (kps[i] + kps[j]) / 2.0

    def ang(p1, p2, flip=False):
        """Ángulo del vector p1→p2 en grados. Si flip=True, invierte el vector."""
        if flip:
            p1, p2 = p2, p1
        return np.degrees(np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0]))

    # ── PITCH — KP1-4 (índices 0-3) ──────────────────────────────────────────
    # Vector mid(trasero) → mid(frontal) representa el eje de pitch.
    # No restar 90°. Calibrar con PITCH_OFFSET en reposo.
    # flip_pitch=True si el giroscopio fue grabado girado 180° en Z.
    pitch = None
    if   vis(0) and vis(1) and vis(2) and vis(3): pitch = ang(mid(0,1), mid(2,3), flip_pitch)
    elif vis(0) and vis(1) and vis(2):             pitch = ang(mid(0,1), kps[2],   flip_pitch)
    elif vis(1) and vis(2) and vis(3):             pitch = ang(kps[1],   mid(2,3), flip_pitch)
    elif vis(0) and vis(2):                        pitch = ang(kps[0],   kps[2],   flip_pitch)
    elif vis(1) and vis(3):                        pitch = ang(kps[1],   kps[3],   flip_pitch)
    elif vis(0) and vis(3):                        pitch = ang(kps[0],   kps[3],   flip_pitch)
    elif vis(1) and vis(2):                        pitch = ang(kps[1],   kps[2],   flip_pitch)

    # ── ROLL — KP5-8 (índices 4-7) ───────────────────────────────────────────
    # v3d: usar vectores por lado en lugar del vector entre puntos medios.
    #
    # GEOMETRÍA:
    #   KP5 (idx=4): punta U arriba-izq  → lado izquierdo
    #   KP6 (idx=5): travesaño arriba-der → lado derecho
    #   KP7 (idx=6): punta U abajo-izq   → lado izquierdo
    #   KP8 (idx=7): travesaño abajo-der  → lado derecho
    #
    # Vector principal: KP5→KP7 (lado izquierdo, punta U — piezas reales).
    # Vector secundario: KP6→KP8 (lado derecho, travesaño).
    # En reposo ambos vectores apuntan hacia abajo en imagen (Y aumenta hacia abajo)
    # → arctan2(-(dy), dx) con dy>0 → arctan2(-dy, 0) = arctan2(negativo, 0) = -90°
    # → fix_angle(-90, +90) = 0° ✓
    #
    # DIAGNÓSTICO DE INVERSIÓN (síntoma: roll ≈ -180° en reposo):
    #   Si el modelo detecta KP5/KP6 con Y > KP7/KP8 (invertidos en imagen),
    #   el vector apunta hacia ARRIBA → arctan2 = +90° → con offset +90 → 180°
    #   → fix_angle(180, +90) = -90° o similar. En ese caso subir ROLL_OFFSET
    #   a -90.0 (o usar --flip-roll si se implementa en próxima versión).
    #
    # FALLBACK si solo un lado está visible: usar ese lado.
    # FALLBACK si ningún lado izq/der: usar par mid(arriba)→mid(abajo) como v3c.
    roll = None
    roll_src = "none"  # para log de diagnóstico

    v5, v6, v7, v8 = vis(4), vis(5), vis(6), vis(7)

    # Calcular ángulo de cada lado disponible
    roll_izq = ang(kps[4], kps[6]) if (v5 and v7) else None  # KP5→KP7
    roll_der = ang(kps[5], kps[7]) if (v6 and v8) else None  # KP6→KP8

    if roll_izq is not None and roll_der is not None:
        # Promedio angular (a través de la media de senos/cosenos para evitar wrap-around)
        rad_izq = np.radians(roll_izq)
        rad_der = np.radians(roll_der)
        avg_sin = (np.sin(rad_izq) + np.sin(rad_der)) / 2.0
        avg_cos = (np.cos(rad_izq) + np.cos(rad_der)) / 2.0
        roll = np.degrees(np.arctan2(avg_sin, avg_cos))
        roll_src = "izq+der"
    elif roll_izq is not None:
        roll = roll_izq
        roll_src = "izq-solo"
    elif roll_der is not None:
        roll = roll_der
        roll_src = "der-solo"
    else:
        # Fallback v3c: mid(arriba)→mid(abajo)
        pa = (kps[4] + kps[5]) / 2.0 if (v5 and v6) else (kps[4] if v5 else (kps[5] if v6 else None))
        pb = (kps[6] + kps[7]) / 2.0 if (v7 and v8) else (kps[6] if v7 else (kps[7] if v8 else None))
        if pa is not None and pb is not None:
            roll = ang(pa, pb, flip=False)
            roll_src = "mid-fallback"

    # Log de diagnóstico de roll (visible con --verbose)
    if verbose_log is not None and roll is not None:
        kp5y = f"{kps[4][1]:.1f}" if v5 else "N/V"
        kp6y = f"{kps[5][1]:.1f}" if v6 else "N/V"
        kp7y = f"{kps[6][1]:.1f}" if v7 else "N/V"
        kp8y = f"{kps[7][1]:.1f}" if v8 else "N/V"
        verbose_log.debug(
            f"[ROLL-DIAG] src={roll_src} raw={roll:.1f}deg "
            f"KP5y={kp5y} KP6y={kp6y} KP7y={kp7y} KP8y={kp8y} "
            f"(esperado: KP5y<KP7y y KP6y<KP8y para vector hacia abajo)")

    # ── YAW — KP9-12 (índices 8-11) ──────────────────────────────────────────
    # flip_yaw=True si el giroscopio fue grabado girado 180° en Z.
    # v3d: el Yaw sigue calculándose pero se marca como no fiable en el panel
    # hasta contar con la segunda cámara.
    yaw = None
    has_izq = vis(8) or vis(11)
    has_der = vis(9) or vis(10)
    if has_izq and has_der:
        mi = mid(8, 11) if (vis(8) and vis(11)) else (kps[8] if vis(8) else kps[11])
        md = mid(9, 10) if (vis(9) and vis(10)) else (kps[9] if vis(9) else kps[10])
        yaw = ang(mi, md, flip_yaw)

    return (fix_angle(roll,  ROLL_OFFSET),
            fix_angle(pitch, PITCH_OFFSET),
            fix_angle(yaw,   YAW_OFFSET))

# ════════════════════════════════════════════════════════════════════════════
#  BUFFER DE HISTORIAL DE ÁNGULOS (para delta)
# ════════════════════════════════════════════════════════════════════════════
# Lista de (timestamp, roll, pitch, yaw). Se limpia automáticamente al superar
# DELTA_BUFFER_S segundos de antigüedad.
angle_history = []   # [(timestamp_float, roll_or_None, pitch_or_None, yaw_or_None)]
angle_history_lock = threading.Lock()

# ── Estado del suavizado EMA (Roll y Pitch) ───────────────────────────────────
# Valores suavizados actuales. Inicializados a None hasta la primera detección.
_ema_roll  = None
_ema_pitch = None

def _smooth_angle(ema_prev, new_val, alpha):
    """
    Media exponencial móvil para ángulos en [-180, 180].
    Maneja el wrap-around: si ema_prev=175° y new_val=-175°, interpola
    en la dirección corta (-10°) no en la larga (+350°).
    alpha=1.0 → sin suavizado (devuelve new_val directamente).

    v3e FIX Bug 16: añadida re-normalización final a [-180, 180].
    En v3d, casos de wrap-around en el límite exacto ±180° podían producir
    valores como -266.8deg porque fix_angle ya había normalizado y la EMA
    acumulaba el error. La normalización final garantiza el rango siempre.
    """
    if ema_prev is None or new_val is None:
        return new_val
    if alpha >= 1.0:
        return new_val
    # Diferencia angular normalizada a [-180, 180] — interpola por el camino corto
    diff = ((new_val - ema_prev + 180.0) % 360.0) - 180.0
    result = ema_prev + alpha * diff
    # Re-normalizar para garantizar que el resultado esté siempre en [-180, 180]
    return ((result + 180.0) % 360.0) - 180.0

def push_angles(roll, pitch, yaw):
    """
    Añade los ángulos actuales al buffer con timestamp actual.
    Limpia entradas más antiguas que DELTA_BUFFER_S.
    """
    now = time.time()
    with angle_history_lock:
        angle_history.append((now, roll, pitch, yaw))
        # Limpiar entradas demasiado antiguas
        cutoff = now - DELTA_BUFFER_S
        while angle_history and angle_history[0][0] < cutoff:
            angle_history.pop(0)

def get_delta(delta_t_s):
    """
    Devuelve (d_roll, d_pitch, d_yaw) como diferencia angular entre el ángulo
    actual y el más antiguo dentro de la ventana delta_t_s.
    Si no hay entrada suficientemente antigua, devuelve (None, None, None).
    Usa delta_angle() para manejar el wrap-around de ±180°.
    """
    now = time.time()
    target_t = now - delta_t_s
    with angle_history_lock:
        if len(angle_history) < 2:
            return None, None, None
        # Entrada más reciente
        _, r_now, p_now, y_now = angle_history[-1]
        # Buscar la entrada más antigua dentro de la ventana
        ref = None
        for entry in angle_history:
            if entry[0] <= target_t:
                ref = entry
            else:
                break
        if ref is None:
            return None, None, None
        _, r_ref, p_ref, y_ref = ref

    d_roll  = delta_angle(r_ref, r_now)
    d_pitch = delta_angle(p_ref, p_now)
    d_yaw   = delta_angle(y_ref, y_now)
    return d_roll, d_pitch, d_yaw

def delta_angle(a_ref, a_now):
    """
    Diferencia angular normalizada a [-180, 180].
    Maneja el wrap-around: si a_ref=170° y a_now=-170°, el delta es -20° (no +340°).
    Devuelve None si alguno de los argumentos es None.
    """
    if a_ref is None or a_now is None:
        return None
    d = a_now - a_ref
    return ((d + 180.0) % 360.0) - 180.0

# ════════════════════════════════════════════════════════════════════════════
#  DIBUJO — panel lateral + keypoints + bboxes
# ════════════════════════════════════════════════════════════════════════════

def _attitude_color(val, dead):
    """
    Devuelve color BGR según magnitud y signo de un ángulo respecto a zona muerta.
    - Dentro de dead: gris neutro
    - Positivo moderado (<45°): verde claro
    - Positivo fuerte (>=45°): verde vivo
    - Negativo moderado: rojo claro
    - Negativo fuerte: rojo vivo
    """
    if val is None:
        return (120, 120, 120)
    a = abs(val)
    if a <= dead:
        return (140, 140, 140)                         # gris — zona muerta
    if val > 0:
        return (20, 220, 20) if a < 45 else (0, 255, 0)   # verde claro → vivo
    return (60, 60, 220) if a < 45 else (0, 0, 255)        # rojo claro → vivo


def _badge(frame, cx, y, text, color_bgr, font, fs):
    """
    Dibuja una píldora de texto centrada en (cx, y).
    Fondo semitransparente del color del indicador, texto blanco.
    """
    (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
    pad_x, pad_y = 8, 4
    x0 = cx - tw // 2 - pad_x
    y0 = y - th - pad_y
    x1 = cx + tw // 2 + pad_x
    y1 = y + pad_y

    # Fondo semitransparente — solo sobre el ROI del badge (no el frame completo)
    # frame[y0:y1, x0:x1] es una VISTA del array, no una copia.
    # addWeighted sobre la vista escribe directamente en el frame.
    h_f, w_f = frame.shape[:2]
    x0c = max(0, x0); y0c = max(0, y0)
    x1c = min(w_f, x1); y1c = min(h_f, y1)
    roi = frame[y0c:y1c, x0c:x1c]
    bg  = roi.copy()                         # copia solo del ROI (~9K px vs 1.5M)
    cv2.rectangle(bg, (0, 0), (x1c - x0c, y1c - y0c), color_bgr, -1)
    cv2.addWeighted(bg, 0.55, roi, 0.45, 0, roi)   # escribe en roi → en frame
    cv2.rectangle(frame, (x0, y0), (x1, y1), color_bgr, 1)
    cv2.putText(frame, text, (cx - tw // 2, y), font, fs, (240, 240, 240), 1)


def _badge_text_pitch(val, dead):
    if val is None:       return "SIN KP"
    if abs(val) <= dead:  return "NIVEL"
    return "ADELANTE" if val > 0 else "ATRAS"


def _badge_text_roll(val, dead):
    if val is None:       return "SIN KP"
    if abs(val) <= dead:  return "NIVELADO"
    return "DER" if val > 0 else "IZQ"


def _badge_text_yaw(val, dead):
    if val is None:      return "N/D"
    if abs(val) <= dead: return "N/D"
    return "HORARIO" if val > 0 else "ANTIHOR"


def _draw_pitch_arc(frame, cx, cy, r, val, dead):
    """
    Arco semicircular superior (abierto hacia abajo).
    El punto de referencia (val=0) está en el centro del arco (parte superior).
    val>0 → punto a la derecha, val<0 → izquierda.
    Eje: de -90° a +90° mapeado al arco de 180° a 360° (en coords imagen).
    """
    # Dibujar arco de fondo (gris oscuro)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 180, 360, (60, 60, 60), 3)
    # Marcas de referencia: 0° (arriba), ±45°, ±90°
    for mark_deg, label in [(-90, ""), (-45, ""), (0, ""), (45, ""), (90, "")]:
        angle_cv = 270 + mark_deg   # 270 = arriba en coords OpenCV
        rad = np.radians(angle_cv)
        mx = int(cx + r * np.cos(rad))
        my = int(cy + r * np.sin(rad))
        cv2.circle(frame, (mx, my), 2, (80, 80, 80), -1)

    # Punto 0° (arriba del arco)
    cv2.circle(frame, (cx, cy - r), 3, (80, 80, 80), -1)

    if val is None:
        return

    # Posición del indicador: val=0→arriba, val=+90→derecha, val=-90→izquierda
    val_clamped = max(-90.0, min(90.0, val))
    angle_cv = 270 + val_clamped        # 270° = arriba en OpenCV
    rad = np.radians(angle_cv)
    px = int(cx + r * np.cos(rad))
    py = int(cy + r * np.sin(rad))

    color = _attitude_color(val, dead)

    # Línea desde centro hasta el punto en el arco
    cv2.line(frame, (cx, cy), (px, py), color, 2)
    # Punto grueso en el arco
    cv2.circle(frame, (px, py), 7, color, -1)
    cv2.circle(frame, (px, py), 8, (0, 0, 0), 1)

    # Línea vertical de referencia (nivel cero)
    cv2.line(frame, (cx, cy), (cx, cy - r), (60, 60, 60), 1)


def _draw_roll_pendulum(frame, cx, cy, r, val, dead):
    """
    Péndulo de Roll: la aguja cuelga del centro y se inclina con el eje.
    val=0  → aguja vertical (arriba)
    val>0  → inclinada a la derecha
    val<0  → inclinada a la izquierda
    Elipse horizonal como "horizonte" de referencia visual.
    """
    # Elipse de referencia (horizonte del eje)
    cv2.ellipse(frame, (cx, cy), (r, r // 3), 0, 0, 360, (50, 50, 50), 1)

    # Línea vertical de referencia
    cv2.line(frame, (cx, cy - r), (cx, cy + r // 4), (60, 60, 60), 1)

    # Marcas ±30° y ±60°
    for mark in [-60, -30, 30, 60]:
        mr = np.radians(-90 + mark)
        mx = int(cx + r * np.cos(mr))
        my = int(cy + r * np.sin(mr))
        cv2.circle(frame, (mx, my), 2, (70, 70, 70), -1)

    if val is None:
        return

    val_clamped = max(-90.0, min(90.0, val))
    angle_cv = -90 + val_clamped   # -90 = apunta arriba en OpenCV
    rad = np.radians(angle_cv)
    px = int(cx + r * np.cos(rad))
    py = int(cy + r * np.sin(rad))

    color = _attitude_color(val, dead)

    # Aguja del péndulo
    cv2.line(frame, (cx, cy), (px, py), color, 3)
    # Punto en el extremo
    cv2.circle(frame, (px, py), 7, color, -1)
    cv2.circle(frame, (px, py), 8, (0, 0, 0), 1)
    # Pivote central
    cv2.circle(frame, (cx, cy), 4, (180, 180, 180), -1)


def _draw_yaw_compass(frame, cx, cy, r, delta_val, dead):
    """
    Brújula de Yaw: siempre muestra el DELTA, no el ángulo absoluto.
    La flecha indica la dirección del giro respecto a los últimos N segundos.
    Color azul para diferenciar de pitch/roll y remarcar que es N/D absoluto.
    """
    # Círculo base
    cv2.circle(frame, (cx, cy), r, (50, 50, 50), 1)

    # Marcas cardinales
    for angle_deg, lbl in [(270, "^"), (0, ">"), (90, "v"), (180, "<")]:
        rad = np.radians(angle_deg)
        mx = int(cx + (r + 8) * np.cos(rad))
        my = int(cy + (r + 8) * np.sin(rad))
        cv2.putText(frame, lbl, (mx - 4, my + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (70, 70, 70), 1)

    # Flecha norte (referencia fija)
    cv2.line(frame, (cx, cy), (cx, cy - r), (60, 60, 60), 1)

    if delta_val is None:
        return

    delta_clamped = max(-180.0, min(180.0, delta_val))
    angle_cv = -90 + delta_clamped
    rad = np.radians(angle_cv)
    px = int(cx + r * np.cos(rad))
    py = int(cy + r * np.sin(rad))

    color = _attitude_color(delta_val, dead)
    # Yaw siempre azul (diferenciador visual) si hay movimiento
    if abs(delta_val) > dead:
        color = (220, 120, 20) if delta_val > 0 else (180, 80, 20)

    cv2.line(frame, (cx, cy), (px, py), color, 3)
    cv2.circle(frame, (px, py), 6, color, -1)
    cv2.circle(frame, (px, py), 7, (0, 0, 0), 1)
    cv2.circle(frame, (cx, cy), 3, (180, 180, 180), -1)


def draw_results(frame, boxes, confs, kps_list, angles_list,
                 fps, scale, pad_x, pad_y, frame_count, delta_t_s,
                 hide_yaw=False, kp_thresh=KP_THRESH, log=None,
                 dead_zone=DEAD_ZONE_DEG):
    """
    v3h: panel de actitud visual con arcos OpenCV.

    Panel derecho dividido en dos zonas:
      1. Cabecera compacta: Det, Conf, FPS, Frame
      2. Tres indicadores visuales: Pitch (arco), Roll (péndulo), Yaw delta (brújula)
         Cada indicador tiene: nombre, gráfico, badge de texto, número pequeño.

    Los indicadores usan zona muerta (dead_zone) para no oscilar con el ruido
    inherente del modelo (~±5-7°). Dentro de la zona muerta todo es gris.
    Los números absolutos se mantienen pequeños bajo cada badge.

    Los bounding boxes y keypoints se dibujan igual que en v3g.
    """
    h, w = frame.shape[:2]
    n_det = len(boxes)

    # Obtener ángulos actuales
    roll_v  = angles_list[0][0] if (angles_list and angles_list[0][0] is not None) else None
    pitch_v = angles_list[0][1] if (angles_list and angles_list[0][1] is not None) else None
    yaw_v   = angles_list[0][2] if (angles_list and angles_list[0][2] is not None) else None

    push_angles(roll_v, pitch_v, yaw_v)
    d_roll, d_pitch, d_yaw = get_delta(delta_t_s)

    # Log verbose FIX 4 (v3g)
    if log is not None and log.isEnabledFor(10):
        dr_s = f"{d_roll:.2f}"  if d_roll  is not None else "None"
        dp_s = f"{d_pitch:.2f}" if d_pitch is not None else "None"
        dy_s = f"{d_yaw:.2f}"  if d_yaw   is not None else "None"
        log.debug(f"[DELTA-DIAG f={frame_count}] d_roll={dr_s}deg "
                  f"d_pitch={dp_s}deg d_yaw={dy_s}deg (ventana={delta_t_s}s)")

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs_hdr = 0.42   # cabecera
    fs_lbl = 0.38   # etiquetas pequeñas bajo los indicadores
    fs_nam = 0.40   # nombre del eje encima del indicador

    # ── PANEL DERECHO ─────────────────────────────────────────────────────────
    # Geometría: panel de 300×350 en la esquina superior derecha
    # Dividido en: cabecera (60px) + zona de indicadores (290px, 3 columnas)
    pw   = 310
    ph   = 340
    px0  = w - pw - 6
    py0  = 6
    # Fondo del panel — solo sobre el ROI del panel (310×340px vs 960×540)
    # Misma técnica que _badge(): vista NumPy, sin copia del frame completo.
    rx0 = max(0, px0 - 2); ry0 = max(0, py0 - 2)
    rx1 = min(w, px0 + pw + 2); ry1 = min(h, py0 + ph + 2)
    roi_panel = frame[ry0:ry1, rx0:rx1]
    bg_panel  = roi_panel.copy()                  # ~316K px vs 1.5M del frame
    cv2.rectangle(bg_panel, (0, 0), (rx1 - rx0, ry1 - ry0), (25, 25, 25), -1)
    cv2.addWeighted(bg_panel, 0.75, roi_panel, 0.25, 0, roi_panel)
    cv2.rectangle(frame, (px0, py0), (px0 + pw, py0 + ph), (70, 70, 70), 1)

    # ── Cabecera compacta ─────────────────────────────────────────────────────
    hdr_h = 58
    cv2.line(frame, (px0, py0 + hdr_h), (px0 + pw, py0 + hdr_h), (70, 70, 70), 1)

    det_color = (0, 220, 0) if n_det > 0 else (0, 0, 220)
    conf_s    = f"{confs[0]:.3f}" if n_det > 0 else "--"
    cv2.putText(frame, f"Det:{n_det}  Conf:{conf_s}",
                (px0 + 8, py0 + 18), font, fs_hdr, det_color, 1)
    cv2.putText(frame, f"FPS:{fps:.1f}  Frame:{frame_count}",
                (px0 + 8, py0 + 38), font, fs_hdr, (180, 180, 180), 1)
    cv2.putText(frame, f"dz=+-{dead_zone:.0f}deg  dt={delta_t_s:.0f}s",
                (px0 + 8, py0 + 55), font, 0.32, (100, 100, 100), 1)

    if n_det == 0:
        cv2.putText(frame, "SIN DETECCION", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # Dibujar indicadores en gris si no hay detección
        roll_v = pitch_v = yaw_v = d_roll = d_pitch = d_yaw = None

    # ── Zona de indicadores: 3 columnas de ancho igual ────────────────────────
    # Cada indicador ocupa pw/3 de ancho.
    # Dentro de cada columna: etiqueta arriba, gráfico en medio, badge+num abajo.
    ind_y0  = py0 + hdr_h + 6   # tope de los indicadores
    col_w   = pw // 3            # ≈103px por columna
    arc_r   = 38                 # radio del arco/péndulo/brújula

    # Centros X de cada columna
    cx_p = px0 + col_w // 2              # Pitch
    cx_r = px0 + col_w + col_w // 2      # Roll
    cx_y = px0 + 2 * col_w + col_w // 2  # Yaw

    # ── PITCH ─────────────────────────────────────────────────────────────────
    arc_cy_p = ind_y0 + arc_r + 20   # centro del arco de pitch
    # Etiqueta
    cv2.putText(frame, "PITCH", (cx_p - 20, ind_y0 + 12),
                font, fs_nam, COLOR_PITCH, 1)
    # Arco
    _draw_pitch_arc(frame, cx_p, arc_cy_p, arc_r, pitch_v, dead_zone)
    # Badge
    badge_y_p = arc_cy_p + arc_r + 22
    badge_txt_p = _badge_text_pitch(pitch_v, dead_zone)
    badge_col_p = _attitude_color(pitch_v, dead_zone)
    _badge(frame, cx_p, badge_y_p, badge_txt_p, badge_col_p, font, 0.36)
    # Número pequeño
    num_p = f"{pitch_v:+.1f}d" if pitch_v is not None else "--"
    cv2.putText(frame, num_p, (cx_p - 22, badge_y_p + 18),
                font, fs_lbl, (130, 130, 130), 1)

    # ── Separador vertical ────────────────────────────────────────────────────
    cv2.line(frame, (px0 + col_w, ind_y0), (px0 + col_w, py0 + ph - 4),
             (55, 55, 55), 1)

    # ── ROLL ──────────────────────────────────────────────────────────────────
    pend_cy_r = ind_y0 + arc_r + 20
    cv2.putText(frame, "ROLL", (cx_r - 16, ind_y0 + 12),
                font, fs_nam, COLOR_ROLL, 1)
    _draw_roll_pendulum(frame, cx_r, pend_cy_r, arc_r, roll_v, dead_zone)
    badge_y_r = pend_cy_r + arc_r + 22
    badge_txt_r = _badge_text_roll(roll_v, dead_zone)
    badge_col_r = _attitude_color(roll_v, dead_zone)
    _badge(frame, cx_r, badge_y_r, badge_txt_r, badge_col_r, font, 0.36)
    num_r = f"{roll_v:+.1f}d" if roll_v is not None else "--"
    cv2.putText(frame, num_r, (cx_r - 22, badge_y_r + 18),
                font, fs_lbl, (130, 130, 130), 1)

    # ── Separador vertical ────────────────────────────────────────────────────
    cv2.line(frame, (px0 + 2 * col_w, ind_y0), (px0 + 2 * col_w, py0 + ph - 4),
             (55, 55, 55), 1)

    # ── YAW ───────────────────────────────────────────────────────────────────
    comp_cy_y = ind_y0 + arc_r + 20
    yaw_label = "YAW" if hide_yaw else "YAW(N/D)"
    cv2.putText(frame, yaw_label, (cx_y - 28, ind_y0 + 12),
                font, 0.34, COLOR_YAW, 1)
    # Siempre dibujamos la brújula con el delta de Yaw (no el absoluto)
    _draw_yaw_compass(frame, cx_y, comp_cy_y, arc_r, d_yaw, dead_zone)
    badge_y_y = comp_cy_y + arc_r + 22
    badge_txt_y = _badge_text_yaw(d_yaw, dead_zone)
    badge_col_y = COLOR_YAW if (d_yaw is not None and abs(d_yaw) > dead_zone) else (100, 100, 100)
    _badge(frame, cx_y, badge_y_y, badge_txt_y, badge_col_y, font, 0.36)
    if not hide_yaw:
        num_y = f"d={d_yaw:+.1f}d" if d_yaw is not None else "--"
        cv2.putText(frame, num_y, (cx_y - 26, badge_y_y + 18),
                    font, fs_lbl, (100, 100, 100), 1)

    # ── Bounding boxes y keypoints ────────────────────────────────────────────
    # Sin cambios respecto a v3g
    if n_det == 0:
        return

    for idx, (box, conf) in enumerate(zip(boxes, confs)):
        x1 = int(np.clip((box[0] - pad_x) / scale, 0, w - 1))
        y1 = int(np.clip((box[1] - pad_y) / scale, 0, h - 1))
        x2 = int(np.clip((box[2] - pad_x) / scale, 0, w - 1))
        y2 = int(np.clip((box[3] - pad_y) / scale, 0, h - 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Giro {conf:.3f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if idx < len(kps_list):
            kps = kps_list[idx]
            for ki in range(NUM_KP):
                kx, ky, ks = kps[ki]
                if ks < kp_thresh: continue
                px = int(np.clip((kx - pad_x) / scale, 0, w - 1))
                py = int(np.clip((ky - pad_y) / scale, 0, h - 1))
                r  = max(4, int(ks * 10))
                cv2.circle(frame, (px, py), r, KP_COLORS[ki], -1)
                cv2.circle(frame, (px, py), r + 1, (0, 0, 0), 1)
                cv2.putText(frame, str(ki + 1), (px + r + 2, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, KP_COLORS[ki], 1)

# ════════════════════════════════════════════════════════════════════════════
#  VERIFICACIÓN AL ARRANQUE
# ════════════════════════════════════════════════════════════════════════════

def verificar_hef(pipe, in_name, log):
    """
    Test de ruido: inyecta imagen aleatoria y comprueba que los tensores de
    confianza tengan algún valor > 0. Si todo es 0, el HEF fue compilado con
    el mismo dataset de entrenamiento y no detectará nada en producción.

    Valores observados en HEFs buenos vs malos:
      calib1: conv71=7   → ✓ OK
      calib2: conv71=24  → ✓ OK
      calib3: conv71=11  → ✓ OK
      estandar: conv71=0 → ✗ MAL — recompilar
    """
    log.info("Verificando HEF con imagen de ruido aleatorio...")
    img  = np.random.randint(0, 255, (MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
    raw  = pipe.infer({in_name: np.expand_dims(img, 0)})
    maxes = {s: int(raw[PREFIX + s][0].max()) for s in ("conv44", "conv58", "conv71")}
    best = max(maxes.values())
    for k, v in maxes.items():
        log.info(f"  {k}: max={v}")
    if best == 0:
        log.error("⚠ PROBLEMA DE CALIBRACIÓN DEL HEF: raw_conf=0 con imagen aleatoria.")
        log.error("  El HEF fue compilado con el mismo dataset de entrenamiento.")
        log.error("  Recompilar con imágenes de calibración de otra escena/dataset.")
        log.error("  Ver documentación: bug_calibracion_hef.md")
        return False
    log.info(f"✓ HEF OK — conf max={best} con ruido aleatorio")
    return True

# ════════════════════════════════════════════════════════════════════════════
#  ESTADO COMPARTIDO (hilo de inferencia ↔ bucle principal)
# ════════════════════════════════════════════════════════════════════════════

latest_frame  = None   # último frame preparado para inferencia
latest_result = None   # último resultado decodificado
frame_lock    = threading.Lock()
result_lock   = threading.Lock()

# ════════════════════════════════════════════════════════════════════════════
#  HILO DE INFERENCIA
# ════════════════════════════════════════════════════════════════════════════

def infer_thread_fn(ng, ng_params, in_name, conf_thresh, kp_scale, args, log, iou_thresh=IOU_THRESH, kp_thresh_val=KP_THRESH):
    """
    Hilo dedicado a inferencia. Lee latest_frame, ejecuta la red Hailo,
    decodifica los 3 strides y escribe en latest_result.

    Separado del bucle principal para que la UI no bloquee la inferencia
    y viceversa.
    """
    global latest_result
    infer_count    = 0
    frames_sin_det = 0

    # Leer configuración de flip desde args (pasados en el arranque)
    flip_pitch = getattr(args, 'flip_pitch', False)
    flip_yaw   = getattr(args, 'flip_yaw',   False)
    log.info(f"flip_pitch={flip_pitch}  flip_yaw={flip_yaw}")
    log.info("Hilo de inferencia iniciado.")
    try:
        with ng.activate(ng_params):
            in_p    = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
            out_u8  = OutputVStreamParams.make(ng, format_type=FormatType.UINT8)
            out_u16 = OutputVStreamParams.make(ng, format_type=FormatType.UINT16)

            # Los tensores de KP son UINT16; el resto UINT8
            kps_names = {PREFIX + n for n in ("conv45", "conv59", "conv72")}
            out_p = {n: p for n, p in out_u8.items()  if n not in kps_names}
            out_p.update({n: p for n, p in out_u16.items() if n in kps_names})
            log.info(f"OutputVStreamParams: {len(out_p)} tensores "
                     f"(6 UINT8 bbox/conf + 3 UINT16 kps)")

            with InferVStreams(ng, in_p, out_p) as pipe:
                verificar_hef(pipe, in_name, log)

                log.info("Bucle de inferencia activo.")
                s8_zeros_warned = False

                while True:
                    with frame_lock:
                        frame_snap = latest_frame
                    if frame_snap is None:
                        time.sleep(0.005)
                        continue

                    img = frame_snap.copy()
                    infer_count += 1

                    try:
                        raw = pipe.infer({in_name: np.expand_dims(img, 0)})
                    except Exception as e:
                        log.error(f"[INFER f={infer_count}] pipe.infer falló: {e}")
                        time.sleep(0.1)
                        continue

                    # Aviso único si s=8 siempre da ceros (bug calibración conocido)
                    if not s8_zeros_warned:
                        if int(raw[PREFIX + "conv44"][0].max()) == 0:
                            log.warning("s=8 raw_conf=0 siempre (bug calibración HEF conocido "
                                        "— ver bug_calibracion_hef.md). Este aviso no se repetirá.")
                            s8_zeros_warned = True

                    # Decodificar los 3 strides
                    all_boxes, all_confs, all_kps = [], [], []
                    conf_maxes = []

                    for stride, b, c, k in [
                        ( 8, "conv43", "conv44", "conv45"),
                        (16, "conv57", "conv58", "conv59"),
                        (32, "conv70", "conv71", "conv72"),
                    ]:
                        key_b = PREFIX + b
                        if key_b not in raw:
                            log.error(f"Tensor {key_b} no encontrado en salida del HEF")
                            continue
                        try:
                            boxes, confs, kps, cm = decode_stride(
                                raw[PREFIX + b][0], raw[PREFIX + c][0], raw[PREFIX + k][0],
                                stride, b, c, k, conf_thresh, kp_scale,
                                log, infer_count, args.verbose)
                            conf_maxes.append(cm)
                            if len(boxes):
                                all_boxes.append(boxes)
                                all_confs.append(confs)
                                all_kps.append(kps)
                        except Exception as e:
                            log.error(f"decode_stride s={stride} f={infer_count}: {e}")
                            log.error(traceback.format_exc())

                    if all_boxes:
                        boxes   = np.concatenate(all_boxes)
                        confs   = np.concatenate(all_confs)
                        kps_all = np.concatenate(all_kps)
                        keep    = nms(boxes, confs, iou_thresh)
                        boxes, confs, kps_all = boxes[keep], confs[keep], kps_all[keep]

                        # Calcular ángulos. En verbose, pasar el logger para diagnóstico
                        # de roll (incluye Y de KP5-8 para detectar inversiones).
                        vlog = log if args.verbose else None
                        angles_raw = [calc_angles(k[:, :2], k[:, 2],
                                                  flip_pitch=flip_pitch,
                                                  flip_yaw=flip_yaw,
                                                  verbose_log=vlog,
                                                  kp_thresh=kp_thresh_val) for k in kps_all]

                        # Aplicar suavizado EMA a Roll y Pitch de la primera detección.
                        # El Yaw no se suaviza: no es fiable con esta perspectiva.
                        global _ema_roll, _ema_pitch
                        angles = []
                        for i, (r_raw, p_raw, y_raw) in enumerate(angles_raw):
                            if i == 0:
                                # Solo suavizamos la detección principal
                                _ema_roll  = _smooth_angle(_ema_roll,  r_raw, ANGLE_SMOOTH_ALPHA)
                                _ema_pitch = _smooth_angle(_ema_pitch, p_raw, ANGLE_SMOOTH_ALPHA)
                                angles.append((_ema_roll, _ema_pitch, y_raw))
                            else:
                                angles.append((r_raw, p_raw, y_raw))

                        r_str = f"{angles[0][0]:.1f}" if angles[0][0] is not None else "N/A"
                        p_str = f"{angles[0][1]:.1f}" if angles[0][1] is not None else "N/A"
                        log.info(f"[f={infer_count}] OK {len(boxes)} det(s) "
                                 f"conf_max={confs.max():.3f} "
                                 f"roll={r_str}deg pitch={p_str}deg")
                        if args.verbose:
                            # Log detallado de KPs para diagnóstico de escala y posición
                            for di, kps in enumerate(kps_all):
                                vis_kps = [(i+1, kps[i,0], kps[i,1], kps[i,2])
                                           for i in range(NUM_KP) if kps[i,2] >= kp_thresh_val]
                                if vis_kps:
                                    xs = [v[1] for v in vis_kps]
                                    ys = [v[2] for v in vis_kps]
                                    log.debug(f"  det[{di}] KPs visibles={len(vis_kps)} "
                                              f"X=[{min(xs):.1f},{max(xs):.1f}] "
                                              f"Y=[{min(ys):.1f},{max(ys):.1f}] "
                                              f"rangos=({max(xs)-min(xs):.1f}, {max(ys)-min(ys):.1f})")
                        frames_sin_det = 0
                    else:
                        # Sin detección: resetear EMA para que no arrastre el
                        # último valor conocido cuando el objeto vuelva a aparecer
                        _ema_roll  = None
                        _ema_pitch = None
                        frames_sin_det += 1
                        if frames_sin_det % 30 == 0:
                            best = max(conf_maxes) if conf_maxes else 0.0
                            log.info(f"[f={infer_count}] 0 detecciones "
                                     f"frames_sin_det={frames_sin_det} "
                                     f"mejor_conf={best:.4f}")
                        boxes   = np.empty((0, 4),        np.float32)
                        confs   = np.empty(0,              np.float32)
                        kps_all = np.empty((0, NUM_KP, 3), np.float32)
                        angles  = []

                    with result_lock:
                        latest_result = (boxes, confs, kps_all, angles)

    except Exception as e:
        log.error(f"Excepción fatal en hilo de inferencia: {e}")
        log.error(traceback.format_exc())

# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    global latest_frame, CONVERT_TO_BGR, FLIP_PITCH, FLIP_YAW, DELTA_T_S

    args         = parse_args()
    log, log_path = setup_logging(args.hef, log_dir_override=args.log_dir)
    kp_scale     = args.kp_scale
    iou_thresh    = args.iou_thresh
    kp_thresh_val = args.kp_thresh
    hide_yaw      = args.hide_yaw
    dead_zone     = args.dead_zone

    # Si se pasa --bgr en CLI, sobreescribe la constante del script
    if args.bgr:
        CONVERT_TO_BGR = True

    # Flags de inversión de eje para giroscopio girado 180° en Z
    if args.flip_pitch:
        FLIP_PITCH = True
    if args.flip_yaw:
        FLIP_YAW = True

    # Ventana temporal del delta (puede ser distinta a la constante por defecto)
    delta_t_s = args.delta_t

    log.info("=" * 60)
    log.info(f"giroscopio_12kp_{VERSION}.py — inicio")
    log.info(f"HEF: {args.hef}")
    log.info(f"Fuente: {'video: ' + args.video if args.video else 'cámara'}")
    log.info(f"CONF_THRESH={args.conf}  KP_THRESH={KP_THRESH}")
    log.info(f"KP_SCALE={kp_scale}  CONVERT_TO_BGR={CONVERT_TO_BGR}")
    log.info(f"FLIP_PITCH={FLIP_PITCH}  FLIP_YAW={FLIP_YAW}")
    log.info(f"DELTA_T_S={delta_t_s}s  DELTA_BUFFER_S={DELTA_BUFFER_S}s")
    log.info(f"IOU_THRESH={iou_thresh}  HIDE_YAW={hide_yaw}  DEAD_ZONE={dead_zone}deg")
    log.info(f"KP_THRESH={kp_thresh_val} (parametrizado)")

    # Log de diagnóstico de rangos esperados de KP por stride
    for st in [8, 16, 32]:
        rango = (kp_scale / 2) * st
        log.info(f"  KP_SCALE={kp_scale} stride={st} → rango máximo KP = ±{rango:.0f}px")

    log.info("=" * 60)

    if not HAVE_HAILO:
        log.error("hailo_platform no disponible.")
        sys.exit(1)

    if not os.path.exists(args.hef):
        log.error(f"HEF no encontrado: {args.hef}")
        sys.exit(1)

    # ── Inicializar Hailo ─────────────────────────────────────────────────────
    log.info("Cargando HEF...")
    hef    = HEF(args.hef)
    target = VDevice()
    cfg    = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng     = target.configure(hef, cfg)[0]
    ng_p   = ng.create_params()
    in_name = hef.get_input_vstream_infos()[0].name
    log.info(f"Input: {in_name}")

    # Leer parámetros de cuantización del HEF y actualizar QUANT + CONF_ALREADY_PROB
    for info in hef.get_output_vstream_infos():
        short = info.name.replace(PREFIX, "")
        if short not in QUANT: continue
        try:
            s  = float(info.quant_info.qp_scale)
            zp = float(info.quant_info.qp_zp)
            QUANT[short] = (s, zp)
            # Formato PROB: zp=0 y scale=1/255 (≈0.003922)
            if short in {"conv44", "conv58", "conv71"} and zp < 100:
                CONF_ALREADY_PROB.add(short)
            log.info(f"  {short}: scale={s:.6f} zp={zp:.1f}"
                     + (" [PROB]" if short in CONF_ALREADY_PROB else ""))
        except AttributeError:
            log.warning(f"  {short}: quant_info no accesible, usando valor de fallback")

    log.info(f"CONF_ALREADY_PROB: {CONF_ALREADY_PROB}")

    # ── Hilo de inferencia ────────────────────────────────────────────────────
    t_infer = threading.Thread(
        target=infer_thread_fn,
        args=(ng, ng_p, in_name, args.conf, kp_scale, args, log, iou_thresh, kp_thresh_val),
        daemon=True)
    t_infer.start()

    # ── Fuente de imagen ──────────────────────────────────────────────────────
    use_video = bool(args.video)
    if use_video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            log.error(f"No se puede abrir: {args.video}")
            sys.exit(1)
        src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        log.info(f"Vídeo: {src_w}×{src_h}  {vid_fps:.1f}fps")
        cam = None
    else:
        if not HAVE_PICAMERA:
            log.error("Picamera2 no disponible y no se especificó --video")
            sys.exit(1)
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"size": (CAM_W, CAM_H), "format": "RGB888"}))
        cam.start()
        time.sleep(1.0)
        src_w, src_h = CAM_W, CAM_H
        cap = None
        log.info(f"Cámara lista: {src_w}×{src_h}")

    # ── Letterbox ─────────────────────────────────────────────────────────────
    # Escala y padding para convertir src_w×src_h → 640×640 con bordes negros
    scale = MODEL_SIZE / max(src_w, src_h)
    nw    = int(src_w * scale)
    nh    = int(src_h * scale)
    pad_x = (MODEL_SIZE - nw) // 2
    pad_y = (MODEL_SIZE - nh) // 2
    log.info(f"Letterbox: scale={scale:.4f} nw={nw} nh={nh} pad_x={pad_x} pad_y={pad_y}")

    # ── VideoWriter ───────────────────────────────────────────────────────────
    # Dos modos de grabacion:
    #   --save <ruta.mp4>     : ruta completa manual (uso desde terminal)
    #   --save-auto <dir/>    : nombre automatico en directorio dado (uso desde wrapper)
    #     Nombre: giroscopio_v3a_<modelo>_<YYYYMMDD_HHMMSS>.mp4
    #     El <modelo> es el mismo que se extrae para el log (dir padre del HEF)
    writer = None
    save_path_efectivo = args.save  # puede ser "" si no se pide grabacion

    if args.save_auto and not args.save:
        # Calcular nombre automatico igual que en setup_logging
        ts_vid  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hef_abs = os.path.abspath(args.hef) if args.hef else ""
        modelo_vid = os.path.basename(os.path.dirname(hef_abs)) if hef_abs else "modelo"
        if not modelo_vid or modelo_vid in (".", ""):
            modelo_vid = "modelo"
        video_dir = args.save_auto
        os.makedirs(video_dir, exist_ok=True)
        save_path_efectivo = os.path.join(
            video_dir, f"giroscopio_v3a_{modelo_vid}_{ts_vid}.mp4")
        log.info(f"Grabacion automatica: {save_path_efectivo}")
    elif args.save and args.save_auto:
        log.warning("--save y --save-auto especificados simultaneamente; "
                    "se usa --save y se ignora --save-auto")

    if save_path_efectivo:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path_efectivo, fourcc, 25, (src_w, src_h))
        log.info(f"Grabando salida en: {save_path_efectivo}" if writer.isOpened()
                 else f"No se pudo abrir VideoWriter: {save_path_efectivo}")

    win_title = f"Giroscopio 12KP {VERSION} — Hailo-8"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    # Escalar ventana según WINDOW_SCALE_FACTOR (restaurado en v3e desde v3b)
    # Factor 1 = nativo (960×540), 2 = FullHD (1920×1080), 3 = 4K (2880×1620)
    cv2.resizeWindow(win_title, src_w * WINDOW_SCALE_FACTOR, src_h * WINDOW_SCALE_FACTOR)
    log.info(f"Ventana: {src_w * WINDOW_SCALE_FACTOR}x{src_h * WINDOW_SCALE_FACTOR} "
             f"(factor={WINDOW_SCALE_FACTOR})")

    fps_t = time.time(); fps_c = 0; fps = 0.0
    frame_count = 0; last_status = time.time()
    log.info("Bucle principal. Pulsa 'q' para salir.")

    try:
        while True:
            # Capturar frame
            if use_video:
                ret, frame_bgr = cap.read()
                if not ret:
                    log.info("Fin del vídeo.")
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                time.sleep(1.0 / max(vid_fps, 1.0))
            else:
                frame_rgb = cam.capture_array()  # entrega RGB888

            frame_count += 1; fps_c += 1

            # Preparar imagen para inferencia cada 2 frames
            # (el hilo de inferencia procesa en background)
            if frame_count % 2 == 0:
                resized = cv2.resize(frame_rgb, (nw, nh))
                lb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
                lb[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

                if CONVERT_TO_BGR:
                    # Conversión explícita si el modelo fue entrenado con BGR
                    img_inf = cv2.cvtColor(lb, cv2.COLOR_RGB2BGR)
                    if frame_count == 2:
                        log.info("Modo BGR activado: convirtiendo RGB→BGR antes de inferencia")
                else:
                    # Por defecto: pasar RGB tal cual (nuevo modelo entrenado con RGB)
                    img_inf = lb

                with frame_lock:
                    latest_frame = img_inf

            # Dibujar resultado (en frame_rgb original para la ventana)
            frame_display = frame_rgb.copy()
            with result_lock:
                result = latest_result

            if result is not None:
                boxes, confs, kps_list, angles_list = result
                draw_results(frame_display, boxes, confs, kps_list, angles_list,
                             fps, scale, pad_x, pad_y, frame_count, delta_t_s,
                             hide_yaw=hide_yaw, kp_thresh=kp_thresh_val, log=log,
                             dead_zone=dead_zone)
            else:
                cv2.putText(frame_display, "Esperando inferencia...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Calcular FPS de la UI (no del hilo de inferencia)
            now = time.time()
            if now - fps_t >= 1.0:
                fps = fps_c / (now - fps_t); fps_c = 0; fps_t = now

            # Status periódico al log (cada 10 s)
            if now - last_status >= 10.0:
                n = len(result[0]) if result is not None else 0
                log.info(f"[STATUS] FPS={fps:.1f} frames={frame_count} "
                         f"detecciones={n} log→{log_path}")
                last_status = now

            # Mostrar y guardar
            cv2.imshow(win_title, frame_display)
            if writer:
                # VideoWriter espera BGR — frame_display es RGB
                writer.write(cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Saliendo por tecla 'q'.")
                break

    except KeyboardInterrupt:
        log.info("Ctrl+C.")
    except Exception as e:
        log.error(f"Excepción en bucle principal: {e}")
        log.error(traceback.format_exc())
    finally:
        if cam:
            try: cam.stop()
            except: pass
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if writer:
            writer.release()
            log.info(f"Vídeo guardado: {args.save}")
        try: target.release()
        except: pass
        log.info(f"Fin. Log en: {log_path}")

if __name__ == "__main__":
    main()
