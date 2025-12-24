import os
import json
import socket
import struct
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from auto_pose.ae import factory

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.disable_eager_execution()

HOST = "localhost"
PORT = 9999
INPUT_SIZE = 128

def send_msg(sock, data):
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_msg(sock):
    raw_len = sock.recv(4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    return sock.recv(msg_len)

# -------- LOAD AAE --------
codebook, _ = factory.build_codebook_from_name(
    "my_autoencoder", "exp_group", return_dataset=True
)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
print("[AAE] Connected to YOLO")

with tf.Session() as sess:
    factory.restore_checkpoint(
        sess,
        tf.train.Saver(),
        "/home/mech/.local/bin/autoencoder_ws/experiments/exp_group/"
        "my_autoencoder_20000_vasti_trainig_4black_backgrounds/checkpoints"
    )
    print("[AAE] Model loaded")

    while True:
        data = recv_msg(sock)
        if not data:
            break

        imgs_hex = json.loads(data.decode())
        Rs = []

        for h in imgs_hex:
            nparr = np.frombuffer(bytes.fromhex(h), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            R = codebook.nearest_rotation(sess, img)
            Rs.append(R.tolist())

        send_msg(sock, json.dumps(Rs).encode())

