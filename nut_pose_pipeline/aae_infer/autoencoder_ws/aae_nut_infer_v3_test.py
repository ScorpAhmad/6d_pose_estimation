import os
import sys
import json
import socket
import struct
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from auto_pose.ae import factory

# Hardening
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.disable_eager_execution()

def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen: return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

# --- SETUP AAE ---
codebook, dataset = factory.build_codebook_from_name("my_autoencoder", "exp_group", return_dataset=True)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9999))

with tf.Session() as sess:
    # the path must be change depend where your checkpoints path
    factory.restore_checkpoint(sess, tf.train.Saver(), "/home/mech/.local/bin/autoencoder_ws/experiments/exp_group/my_autoencoder_20000_vasti_trainig_4black_backgrounds/checkpoints")
    print("AAE Engine Ready.")

    while True:
        data = recv_msg(client)
        if not data: break
        
        hex_crops = json.loads(data.decode('utf-8'))
        batch_Rs = []

        for h in hex_crops:
            # Decode hex back to bytes then to image
            nparr = np.frombuffer(bytes.fromhex(h), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                img_input = cv2.cvtColor(cv2.resize(img, (128,128)), cv2.COLOR_BGR2RGB)
                R = codebook.nearest_rotation(sess, img_input)
                batch_Rs.append(R.tolist())

        # Send rotations back as JSON
        send_msg(client, json.dumps(batch_Rs).encode('utf-8'))
