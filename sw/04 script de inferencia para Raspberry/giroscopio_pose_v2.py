#!/usr/bin/env python3
"""
giroscopio_pose_v2.py — Inferencia YOLOv8n-Pose en Hailo-8
HEF: giroscopio_pose_v2.hef  (compilado con DFC 3.33.0 para hailo8l)

Tensores UINT8 con dequantización: float = (uint8 - zp) * scale
  conv44/58/71 (confianza): zp=255 → valores negativos = baja confianza
"""

import cv2
import numpy as np
import threading
import time
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                             ConfigureParams, InputVStreamParams,
                             OutputVStreamParams, FormatType, InferVStreams)

# ── Configuración ────────────────────────────────────────────────────────────
HEF_PATH    = "/home/ai/giroscopio_pose_v2.hef"
CONF_THRESH = 0.25
KP_THRESH   = 0.30
NUM_KP      = 9
IOU_THRESH  = 0.45
PREFIX      = "giroscopio_pose_v2/"

# Offsets de corrección reposo → 0º
ROLL_OFFSET  = 180.0
PITCH_OFFSET = 90.0
YAW_OFFSET   = 180.0

# Parámetros de dequantización (scale, zero_point) por tensor
QUANT = {
    "conv43": (0.136709,  93.0),   # bbox  stride8
    "conv44": (0.118598, 255.0),   # conf  stride8
    "conv45": (0.062451, 174.0),   # kps   stride8
    "conv57": (0.113493, 128.0),   # bbox  stride16
    "conv58": (0.095957, 255.0),   # conf  stride16
    "conv59": (0.046742, 168.0),   # kps   stride16
    "conv70": (0.090731, 145.0),   # bbox  stride32
    "conv71": (0.257394, 246.0),   # conf  stride32
    "conv72": (0.057268, 111.0),   # kps   stride32
}

# Colores keypoints
KP_COLORS = [
    (0,   255, 255),  # 0 centro_base       — amarillo
    (0,   0,   255),  # 1 ext_der_rojo       — rojo
    (0,   0,   200),  # 2 ext_izq_rojo       — rojo oscuro
    (0,   255, 0  ),  # 3 frente_octogono    — verde
    (0,   200, 0  ),  # 4 trasero_octogono   — verde oscuro
    (255, 0,   0  ),  # 5 pivote_der         — azul
    (200, 0,   0  ),  # 6 pivote_izq         — azul oscuro
    (255, 255, 0  ),  # 7 esquina_der_azul   — cian
    (255, 200, 0  ),  # 8 esquina_izq_azul   — cian oscuro
]

# ── Utilidades ───────────────────────────────────────────────────────────────
def dequant(tensor_uint8, name):
    scale, zp = QUANT[name]
    return (tensor_uint8.astype(np.float32) - zp) * scale

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def softmax_dfl(x, reg_max=16):
    """DFL: softmax sobre reg_max bins → valor esperado."""
    x = x.reshape(*x.shape[:-1], 4, reg_max)
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    sm = ex / ex.sum(axis=-1, keepdims=True)
    bins = np.arange(reg_max, dtype=np.float32)
    return (sm * bins).sum(axis=-1)

def decode_stride(raw_bbox, raw_conf, raw_kps, stride,
                  b_name, c_name, k_name):
    """Decodifica un nivel de escala. Devuelve (boxes_xyxy, confs, keypoints)."""
    H, W = raw_conf.shape[:2]

    # Dequantizar
    bbox_f = dequant(raw_bbox, b_name)
    conf_f = dequant(raw_conf, c_name)
    kps_f  = dequant(raw_kps,  k_name)

    # Confianza — conv44/58/71 tienen zp=255 → resultado es negativo → sigmoid
    conf_map = sigmoid(conf_f[..., 0])

    mask = conf_map > CONF_THRESH
    if not mask.any():
        return np.empty((0,4)), np.empty(0), np.empty((0, NUM_KP, 3))

    ys, xs = np.where(mask)
    confs   = conf_map[ys, xs]

    # BBox via DFL
    bbox_sel = bbox_f[ys, xs]  # (N, 64)
    ltrb = softmax_dfl(bbox_sel) * stride  # (N, 4)
    cx = (xs + 0.5) * stride
    cy = (ys + 0.5) * stride
    x1 = cx - ltrb[:, 0]
    y1 = cy - ltrb[:, 1]
    x2 = cx + ltrb[:, 2]
    y2 = cy + ltrb[:, 3]
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Keypoints — transformación correcta YOLOv8-Pose
    kps_sel = kps_f[ys, xs].reshape(-1, NUM_KP, 3)
    kp_x_raw = kps_sel[..., 0]
    kp_y_raw = kps_sel[..., 1]
    kp_x = (1.0 / (1.0 + np.exp(-kp_x_raw)) * 10.0 - 5.0 + xs[:, None]) * stride
    kp_y = (1.0 / (1.0 + np.exp(-kp_y_raw)) * 10.0 - 5.0 + ys[:, None]) * stride
    kp_score = sigmoid(kps_sel[..., 2])
    keypoints = np.concatenate([
        kp_x[..., None], kp_y[..., None], kp_score[..., None]
    ], axis=-1)

    return boxes, confs, keypoints

def nms(boxes, scores, iou_thresh=IOU_THRESH):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep

def calc_angles(kps, scores):
    """Calcula Roll, Pitch, Yaw en grados a partir de los 9 keypoints."""
    def ang(p1, p2, s1, s2, thresh=KP_THRESH):
        if s1 > thresh and s2 > thresh:
            return np.degrees(np.arctan2(-(p2[1]-p1[1]), p2[0]-p1[0]))
        return None

    roll  = ang(kps[2], kps[1], scores[2], scores[1])
    if roll is None:
        roll = ang(kps[6], kps[5], scores[6], scores[5])

    pitch = ang(kps[3], kps[4], scores[3], scores[4])
    if pitch is None and scores[0] > KP_THRESH and scores[3] > KP_THRESH:
        pitch = ang(kps[0], kps[3], scores[0], scores[3])

    yaw = None
    if scores[7] > KP_THRESH and scores[0] > KP_THRESH:
        yaw = np.degrees(np.arctan2(-(kps[7][1]-kps[0][1]),
                                     kps[7][0]-kps[0][0]))
    elif scores[8] > KP_THRESH and scores[0] > KP_THRESH:
        yaw = np.degrees(np.arctan2(-(kps[8][1]-kps[0][1]),
                                     kps[8][0]-kps[0][0])) - 90

    # Corrección offset reposo → 0º y normalización a [-180, 180]
    def fix(v, offset):
        if v is None:
            return None
        v = v + offset
        while v >  180: v -= 360
        while v < -180: v += 360
        return v

    return fix(roll, ROLL_OFFSET), fix(pitch, PITCH_OFFSET), fix(yaw, YAW_OFFSET)

def draw_results(frame, boxes, confs, keypoints_list, angles_list, fps):
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    pad_y = (640 - nh) // 2
    pad_x = (640 - nw) // 2
    sx, sy = 1/scale, 1/scale

    max_conf = float(confs.max()) if len(confs) > 0 else 0.0
    roll_d = pitch_d = yaw_d = "--"

    for box, conf, kps, (roll, pitch, yaw) in zip(boxes, confs, keypoints_list, angles_list):
        x1,y1,x2,y2 = ((box - [pad_x,pad_y,pad_x,pad_y]) * [sx,sy,sx,sy]).astype(int)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        for i, (kp, color) in enumerate(zip(kps, KP_COLORS)):
            kx, ky, ks = kp
            if ks > KP_THRESH:
                px, py = int((kx-pad_x)*sx), int((ky-pad_y)*sy)
                cv2.circle(frame, (px,py), 5, color, -1)
                cv2.putText(frame, str(i), (px+4,py-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if roll  is not None: roll_d  = f"{roll:+.1f}"
        if pitch is not None: pitch_d = f"{pitch:+.1f}"
        if yaw   is not None: yaw_d   = f"{yaw:+.1f}"

    # Panel fijo esquina superior derecha
    pw, ph = 200, 140
    px0 = w - pw - 10
    py0 = 10
    cv2.rectangle(frame, (px0, py0), (px0+pw, py0+ph), (70,70,70), -1)
    cv2.rectangle(frame, (px0, py0), (px0+pw, py0+ph), (160,160,160), 1)
    for i, (txt, col) in enumerate([
        (f"FPS:   {fps:.1f}",     (0,255,255)),
        (f"Conf:  {max_conf:.2f}",(200,200,200)),
        (f"Roll:  {roll_d} deg",  (0,0,255)),
        (f"Pitch: {pitch_d} deg", (0,255,0)),
        (f"Yaw:   {yaw_d} deg",   (255,128,0)),
    ]):
        cv2.putText(frame, txt, (px0+8, py0+24+i*23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    cv2.putText(frame, f"Det: {len(boxes)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    return frame

# ── Pipeline Hailo ───────────────────────────────────────────────────────────
hef    = HEF(HEF_PATH)
target = VDevice()
cfg    = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
ng     = target.configure(hef, cfg)[0]
in_p   = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
out_p  = OutputVStreamParams.make(ng, format_type=FormatType.UINT8)
infer_pipeline = InferVStreams(ng, in_p, out_p)
ng_params      = ng.create_params()

in_name = hef.get_input_vstream_infos()[0].name

# Estado compartido entre hilos
latest_frame  = None
latest_result = None
frame_lock    = threading.Lock()
result_lock   = threading.Lock()
frame_count   = 0

def infer_thread():
    global latest_result
    with infer_pipeline as pipe:
        with ng.activate(ng_params):
            while True:
                with frame_lock:
                    if latest_frame is None:
                        time.sleep(0.005)
                        continue
                    img = latest_frame.copy()

                inp = {in_name: np.expand_dims(img, 0)}
                raw = pipe.infer(inp)

                # Decodificar las 3 escalas
                all_boxes, all_confs, all_kps = [], [], []
                for stride, b_name, c_name, k_name in [
                    (8,  "conv43","conv44","conv45"),
                    (16, "conv57","conv58","conv59"),
                    (32, "conv70","conv71","conv72"),
                ]:
                    boxes, confs, kps = decode_stride(
                        raw[PREFIX+b_name][0],
                        raw[PREFIX+c_name][0],
                        raw[PREFIX+k_name][0],
                        stride,
                        b_name, c_name, k_name
                    )
                    if len(boxes):
                        all_boxes.append(boxes)
                        all_confs.append(confs)
                        all_kps.append(kps)
                if all_boxes:
                    boxes  = np.concatenate(all_boxes)
                    confs  = np.concatenate(all_confs)
                    kps    = np.concatenate(all_kps)
                    keep   = nms(boxes, confs)
                    boxes, confs, kps = boxes[keep], confs[keep], kps[keep]
                    angles = [calc_angles(k[:,:2], k[:,2]) for k in kps]
                else:
                    boxes = np.empty((0,4))
                    confs = np.empty(0)
                    kps   = np.empty((0,NUM_KP,3))
                    angles = []

                with result_lock:
                    latest_result = (boxes, confs, kps, angles)

# Arrancar hilo de inferencia
t = threading.Thread(target=infer_thread, daemon=True)
t.start()

# ── Cámara y display ─────────────────────────────────────────────────────────
cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"size": (960, 540), "format": "RGB888"}))
cam.start()
time.sleep(1)

cv2.namedWindow("Giroscopio — Hailo-8", cv2.WINDOW_NORMAL)

fps_t  = time.time()
fps_c  = 0
fps    = 0.0

print("INFO] Corriendo — pulsa 'q' para salir.")

try:
    while True:
        frame_rgb = cam.capture_array()
        frame_bgr = frame_rgb  # Picamera2 RGB888 ya es compatible con display

        # Preparar imagen para inferencia cada 2 frames
        fps_c += 1
        if fps_c % 2 == 0:
            h, w = frame_bgr.shape[:2]
            # Letterbox: escalar manteniendo ratio y rellenar con negro
            scale = 640 / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(frame_bgr, (nw, nh))
            img_inf = np.zeros((640, 640, 3), dtype=np.uint8)
            pad_y = (640 - nh) // 2
            pad_x = (640 - nw) // 2
            img_inf[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
            img_inf = cv2.cvtColor(img_inf, cv2.COLOR_RGB2BGR)
            with frame_lock:
                latest_frame = img_inf

        # Dibujar último resultado disponible
        with result_lock:
            result = latest_result

        display = frame_bgr.copy()
        if result is not None:
            boxes, confs, kps, angles = result
            display = draw_results(display, boxes, confs, kps, angles, fps)
        else:
            cv2.putText(display, f"Det: 0", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # FPS
        now = time.time()
        if now - fps_t >= 1.0:
            fps   = fps_c / (now - fps_t)
            fps_c = 0
            fps_t = now
        cv2.putText(display, f"FPS: {fps:.1f}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Giroscopio — Hailo-8", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()
    target.release()
