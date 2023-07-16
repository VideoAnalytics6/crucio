import json
import os
import socket
import struct

import torch

from autoencoder.dataset import IMAGE_DIR, IMAGE_EXT
from autoencoder.network2d import get_networks
from autoencoder.util import (check_work_dir, convert_image_to_tensor,
                              GPU_ENABLED, save_compressed_data, YUV_ENABLED)

check_work_dir()
with open('config.json', 'r') as f:
    config = json.load(f)
server_ip = config['host']
server_port = config['port']
datalength = config['datalength']
head = config['filehead']
chunk_size = config['bufsize']
encoder, decoder = get_networks('eval', True)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

i = 0
for file_name in os.listdir(IMAGE_DIR):
    if i >= datalength:
        break
    if file_name.endswith(IMAGE_EXT):
        file_path = os.path.join(IMAGE_DIR, file_name)
        img_tensor = convert_image_to_tensor(
            file_path, YUV_ENABLED, GPU_ENABLED).unsqueeze(0)
        with torch.no_grad():
            compressed_data = encoder(img_tensor)
        data_path = save_compressed_data(file_path, compressed_data)
        print(f'Encoding {file_path} to {data_path}')
        file_size = os.path.getsize(data_path)
        file_info = struct.pack(head, os.path.splitext(
            file_name)[0].encode(), file_size)
        client_socket.send(file_info)
        with open(data_path, "rb") as f:
            client_socket.sendall(f.read())
        print(f'Send {data_path} to server')
        i += 1
client_socket.close()
