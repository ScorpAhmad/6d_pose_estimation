import cv2
import socket
import struct
import json
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "/home/mech/nut_pose_estemation/models_of_YoloV12-seg/best.pt"
HOST = "localhost"
PORT = 9999
CONF_TH = 0.35
INPUT_SIZE = 128
# ----------------------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

# -------- SOCKET SERVER --------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print("[YOLO] Waiting for AAE engine...")
conn, _ = server.accept()
print("[YOLO] AAE connected")

def send_msg(sock, data: bytes):
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_msg(sock):
    raw_len = sock.recv(4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    return sock.recv(msg_len)

# -------- PREPROCESS FOR AAE --------
def preprocess_mask_crop(frame, mask):
    mask = (mask > 0.5).astype(np.uint8)
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return None, None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    obj = frame.copy()
    obj[mask == 0] = 0
    crop = obj[y1:y2, x1:x2]

    h, w = crop.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    resized = cv2.resize(crop, (int(w*scale), int(h*scale)))

    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    yoff = (INPUT_SIZE - resized.shape[0]) // 2
    xoff = (INPUT_SIZE - resized.shape[1]) // 2
    canvas[yoff:yoff+resized.shape[0], xoff:xoff+resized.shape[1]] = resized

    center = ((x1+x2)//2, (y1+y2)//2)
    return canvas, center

# -------- AXIS DRAW (rotation only, for now) --------
def draw_axis_2d(img, R, center):
    axis = np.float32([[30,0,0],[0,30,0],[0,0,-30],[0,0,0]])
    pts = (axis @ R.T)[:, :2]
    pts = [(int(p[0]+center[0]), int(p[1]+center[1])) for p in pts]
    cv2.line(img, pts[3], pts[0], (0,0,255), 2)
    cv2.line(img, pts[3], pts[1], (0,255,0), 2)
    cv2.line(img, pts[3], pts[2], (255,0,0), 2)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONF_TH, verbose=False)[0]
    payload_imgs = []
    centers = []

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()

        for mask in masks:
            img_aae, center = preprocess_mask_crop(frame, mask)
            if img_aae is None:
                continue

            _, enc = cv2.imencode(".jpg", img_aae)
            payload_imgs.append(enc.tobytes().hex())
            centers.append(center)

    if payload_imgs:
        send_msg(conn, json.dumps(payload_imgs).encode())
        response = recv_msg(conn)

        if response:
            Rs = json.loads(response.decode())
            for R, ctr in zip(Rs, centers):
                draw_axis_2d(frame, np.array(R), ctr)

    cv2.imshow("YOLO + AAE 6D Pose (Rotation)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
conn.close()

