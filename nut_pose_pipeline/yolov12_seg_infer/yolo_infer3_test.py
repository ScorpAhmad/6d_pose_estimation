import cv2
import socket
import struct
import json
import numpy as np
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "/home/mech/nut_pose_estemation/models_of_YoloV12-seg/best.pt"
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

# Socket Server Setup
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('localhost', 9999))
server.listen(1)
print("Waiting for AAE Engine...")
conn, addr = server.accept()

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen: return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

def draw_axis(img, R, center):
    scale = 45
    axis_3d = np.float32([[scale,0,0], [0,scale,0], [0,0,-scale], [0,0,0]]).reshape(-1,3)
    projected = (axis_3d @ R.T)
    pts_2d = [(int(p[0] + center[0]), int(p[1] + center[1])) for p in projected]
    cv2.line(img, pts_2d[3], pts_2d[0], (0,0,255), 2) # X
    cv2.line(img, pts_2d[3], pts_2d[1], (0,255,0), 2) # Y
    cv2.line(img, pts_2d[3], pts_2d[2], (255,0,0), 2) # Z

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(frame, conf=0.35, verbose=False)[0]
    
    crops_to_send = []
    centers = []

    if results.boxes:
        for box in results.boxes.xyxy.cpu().numpy().astype(int):
            crop = frame[box[1]:box[3], box[0]:box[2]]
            if crop.size > 0:
                _, img_enc = cv2.imencode('.jpg', crop)
                # Convert to list of hex strings or base64 to avoid pickle issues
                crops_to_send.append(img_enc.tobytes().hex())
                centers.append(((box[0]+box[2])//2, (box[1]+box[3])//2))

    if crops_to_send:
        # Send as JSON string to avoid NumPy DType mismatch
        payload = json.dumps(crops_to_send).encode('utf-8')
        send_msg(conn, payload)

        # Receive rotations
        response = recv_msg(conn)
        if response:
            all_Rs = json.loads(response.decode('utf-8'))
            for R_list, ctr in zip(all_Rs, centers):
                draw_axis(frame, np.array(R_list), ctr)

    cv2.imshow("6D Pose estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
conn.close()
