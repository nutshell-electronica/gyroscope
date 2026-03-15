#!/usr/bin/env python3
"""
YOLOv8s Pose Estimation – Hailo-8 – Raspberry Pi 5
---------------------------------------------------
Uso:
  DISPLAY=:0 python3 yolov8_pose_hailo.py
  DISPLAY=:0 python3 yolov8_pose_hailo.py --save output.mp4
"""

import argparse
import os
import time
import threading
import numpy as np
import cv2

os.environ.setdefault("DISPLAY", ":0")
os.environ.pop("WAYLAND_DISPLAY", None)

from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType,
)

# --------------------------------------------------------------------------- #
#  Constantes COCO pose (17 keypoints)                                        #
# --------------------------------------------------------------------------- #
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
KP_COLOR   = (0, 255, 0)
BONE_COLOR = (0, 165, 255)
BOX_COLOR  = (0, 200, 255)

# Umbrales — confianza en escala RAW (el h8 ya aplica sigmoid internamente)
CONF_THRESH = 0.40   # para conv44/58/71 en escala lineal [0..1]
KP_THRESH   = 0.30   # para scores de keypoints (también lineales)

# Tensores del HEF yolov8s_pose_h8
# conv43/57/70 → DFL bbox (64 canales = 4 × 16 bins)
# conv44/58/71 → confianza ya en [0,1] (NO aplicar sigmoid)
# conv45/59/72 → keypoints 17×3 (x,y ya normalizados, score en logit → sigmoid)
BOX_KEYS  = ["yolov8s_pose/conv43", "yolov8s_pose/conv57", "yolov8s_pose/conv70"]
CONF_KEYS = ["yolov8s_pose/conv44", "yolov8s_pose/conv58", "yolov8s_pose/conv71"]
KPS_KEYS  = ["yolov8s_pose/conv45", "yolov8s_pose/conv59", "yolov8s_pose/conv72"]
STRIDES   = [8, 16, 32]


# --------------------------------------------------------------------------- #
#  Argumentos                                                                  #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hef",    default="/usr/share/hailo-models/yolov8s_pose_h8.hef")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--save",   default="")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Cámara                                                                      #
# --------------------------------------------------------------------------- #
def init_camera(width, height):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    ))
    cam.start()
    time.sleep(1.5)
    print(f"[INFO] Cámara lista {width}x{height}")
    return cam


# --------------------------------------------------------------------------- #
#  Hailo – pipeline persistente                                                #
# --------------------------------------------------------------------------- #
class HailoInference:
    def __init__(self, hef_path):
        print(f"[INFO] Cargando HEF: {hef_path}")
        self.hef    = HEF(hef_path)
        self.target = VDevice()
        cfg  = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        ng   = self.target.configure(self.hef, cfg)[0]
        self.ng   = ng
        self.ng_p = ng.create_params()
        self.in_p  = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        self.out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        info = self.hef.get_input_vstream_infos()[0]
        self.input_name      = info.name
        self.model_h, self.model_w = info.shape[0], info.shape[1]
        # Pipeline abierto permanentemente
        self._pipe = InferVStreams(self.ng, self.in_p, self.out_p)
        self._pipe.__enter__()
        self._ctx  = self.ng.activate(self.ng_p)
        self._ctx.__enter__()
        print(f"[INFO] Hailo listo – input {self.model_w}×{self.model_h}")

    def infer(self, frame_bgr):
        img = cv2.resize(frame_bgr, (self.model_w, self.model_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = {self.input_name: np.expand_dims(img, 0).astype(np.uint8)}
        return self._pipe.infer(inp)

    def close(self):
        try:
            self._ctx.__exit__(None, None, None)
            self._pipe.__exit__(None, None, None)
        except Exception:
            pass
        self.target.release()


# --------------------------------------------------------------------------- #
#  Post-procesado                                                              #
# --------------------------------------------------------------------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def decode_stride(raw_box, raw_conf, raw_kps, stride, orig_w, orig_h, model_w, model_h):
    """
    raw_conf ya está en [0,1] — el HEF h8 aplica sigmoid internamente.
    raw_box  son los bins DFL crudos (sin softmax).
    raw_kps  x,y están en espacio de grilla; score es logit → aplicar sigmoid.
    """
    REG_MAX  = 16
    conf_map = raw_conf[..., 0]          # (H, W) valores en [0,1]
    mask     = conf_map > CONF_THRESH
    if not mask.any():
        return []

    ys, xs = np.where(mask)

    # DFL → ltrb en unidades de stride
    reg      = raw_box[ys, xs].reshape(-1, 4, REG_MAX)
    reg_exp  = np.exp(reg - reg.max(axis=-1, keepdims=True))
    reg_soft = reg_exp / reg_exp.sum(axis=-1, keepdims=True)
    ltrb     = (reg_soft * np.arange(REG_MAX, dtype=np.float32)).sum(axis=-1)

    # BBox en píxeles del modelo
    cx = (xs + 0.5) * stride;  cy = (ys + 0.5) * stride
    x1 = cx - ltrb[:, 0] * stride;  y1 = cy - ltrb[:, 1] * stride
    x2 = cx + ltrb[:, 2] * stride;  y2 = cy + ltrb[:, 3] * stride

    # Keypoints
    kps_raw = raw_kps[ys, xs].reshape(-1, 17, 3)
    kp_x    = (kps_raw[..., 0] * 2.0 + xs[:, None]) * stride
    kp_y    = (kps_raw[..., 1] * 2.0 + ys[:, None]) * stride
    kp_s    = sigmoid(kps_raw[..., 2])   # score sí necesita sigmoid

    # Escalar al frame original
    sx, sy = orig_w / model_w, orig_h / model_h
    confs  = conf_map[mask]

    results = []
    for i in range(len(ys)):
        results.append({
            "bbox": (
                int(np.clip(x1[i]*sx, 0, orig_w-1)),
                int(np.clip(y1[i]*sy, 0, orig_h-1)),
                int(np.clip(x2[i]*sx, 0, orig_w-1)),
                int(np.clip(y2[i]*sy, 0, orig_h-1)),
            ),
            "conf": float(confs[i]),
            "keypoints": [
                (int(np.clip(kp_x[i,k]*sx, 0, orig_w-1)),
                 int(np.clip(kp_y[i,k]*sy, 0, orig_h-1)),
                 float(kp_s[i, k]))
                for k in range(17)
            ],
        })
    return results


def nms(dets, iou_thresh=0.45):
    if not dets:
        return []
    boxes  = np.array([d["bbox"]  for d in dets], dtype=np.float32)
    scores = np.array([d["conf"]  for d in dets], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2-x1).clip(0) * (y2-y1).clip(0)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2-xx1).clip(0) * (yy2-yy1).clip(0)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return [dets[k] for k in keep]


def decode_pose(outputs, orig_w, orig_h, model_w, model_h):
    all_dets = []
    for bk, ck, kk, stride in zip(BOX_KEYS, CONF_KEYS, KPS_KEYS, STRIDES):
        all_dets.extend(
            decode_stride(outputs[bk][0], outputs[ck][0], outputs[kk][0],
                          stride, orig_w, orig_h, model_w, model_h)
        )
    return nms(all_dets)


# --------------------------------------------------------------------------- #
#  Dibujado de keypoints y esqueleto                                           #
# --------------------------------------------------------------------------- #
def draw_pose(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        kps = det["keypoints"]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f"{det['conf']:.2f}",
                    (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)

        # Esqueleto
        for i, j in SKELETON:
            xi, yi, si = kps[i]
            xj, yj, sj = kps[j]
            if si > KP_THRESH and sj > KP_THRESH:
                cv2.line(frame, (xi, yi), (xj, yj), BONE_COLOR, 2, cv2.LINE_AA)

        # Keypoints
        for kx, ky, ks in kps:
            if ks > KP_THRESH:
                cv2.circle(frame, (kx, ky), 5, KP_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, (kx, ky), 5, (0, 0, 0), 1, cv2.LINE_AA)  # borde negro

    return frame


# --------------------------------------------------------------------------- #
#  Hilo de inferencia                                                          #
# --------------------------------------------------------------------------- #
class InferenceThread(threading.Thread):
    def __init__(self, hailo, orig_w, orig_h):
        super().__init__(daemon=True)
        self.hailo       = hailo
        self.orig_w      = orig_w
        self.orig_h      = orig_h
        self.input_frame = None
        self.detections  = []
        self.running     = True
        self.lock        = threading.Lock()
        self.event       = threading.Event()

    def submit(self, frame):
        with self.lock:
            self.input_frame = frame.copy()
        self.event.set()

    def run(self):
        while self.running:
            self.event.wait(timeout=1.0)
            self.event.clear()
            with self.lock:
                frame = self.input_frame
            if frame is None:
                continue
            try:
                outputs = self.hailo.infer(frame)
                dets    = decode_pose(outputs, self.orig_w, self.orig_h,
                                      self.hailo.model_w, self.hailo.model_h)
                with self.lock:
                    self.detections = dets
            except Exception as e:
                print(f"[WARN] Inferencia: {e}")

    def get_detections(self):
        with self.lock:
            return list(self.detections)

    def stop(self):
        self.running = False
        self.event.set()


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    cam   = init_camera(args.width, args.height)
    hailo = HailoInference(args.hef)

    infer_t = InferenceThread(hailo, args.width, args.height)
    infer_t.start()

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 25, (args.width, args.height))
        print(f"[INFO] Guardando en: {args.save}")

    cv2.namedWindow("YOLOv8 Pose – Hailo-8", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Pose – Hailo-8", args.width, args.height)
    print("[INFO] Corriendo — pulsa 'q' para salir.")

    fps_t = time.time()
    fps_n = 0
    fps_v = 0.0
    fc    = 0

    try:
        while True:
            frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)

            # Enviar al hilo de inferencia cada 2 frames
            if fc % 2 == 0:
                infer_t.submit(frame)
            fc += 1

            # Dibujar sobre copia del frame
            display    = frame.copy()
            detections = infer_t.get_detections()
            draw_pose(display, detections)

            # FPS overlay
            fps_n += 1
            elapsed = time.time() - fps_t
            if elapsed >= 1.0:
                fps_v = fps_n / elapsed
                fps_n = 0
                fps_t = time.time()
            cv2.putText(display,
                        f"FPS: {fps_v:.1f}  Personas: {len(detections)}",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLOv8 Pose – Hailo-8", display)
            if writer:
                writer.write(display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Detenido.")
    finally:
        infer_t.stop()
        infer_t.join(timeout=3)
        cam.stop()
        hailo.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Listo.")


if __name__ == "__main__":
    main()
