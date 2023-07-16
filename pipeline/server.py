import json
import os
import socket
import struct

import torch

from autoencoder.dataset import IMAGE_DIR, IMAGE_EXT
from autoencoder.network2d import get_networks
from autoencoder.util import (YUV_ENABLED, check_work_dir,
                              convert_image_to_tensor, convert_tensor_to_image,
                              load_compressed_data)
# Server CUDA and PyTorch versions are too low
# from dnn_model.faster_rcnn import test_faster_rcnn
# from dnn_model.fcos import test_fcos
# from dnn_model.ssd import test_ssd
from dnn_model.util import convert_tensor_to_numpy
from dnn_model.yolov5 import test_yolov5

check_work_dir()
with open('config.json', 'r') as f:
    config = json.load(f)
server_ip = config['host']
server_port = config['port']
dnn_task = config['task']
datalength = config['datalength']
head = config['filehead']
chunk_size = config['bufsize']
file_info_size = struct.calcsize(head)
encoder, decoder = get_networks('eval', True)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.settimeout(20)
try:
    server_socket.listen(1)
    print('Wait for client to connect...')
    conn, addr = server_socket.accept()
    print('Client is connected:', addr)
except socket.timeout:
    print('No connection received within 20s, receiver exits')
    server_socket.close()
    exit(0)

i = 0
while i < datalength:
    info_size = 0
    while info_size < file_info_size:
        file_info = conn.recv(file_info_size-info_size)
        info_size += len(file_info)
    assert info_size == file_info_size
    file_name, file_size = struct.unpack(head, file_info)
    base_path = os.path.join(
        IMAGE_DIR, file_name.decode().replace('\x00', ''))
    file_path = base_path+'.pkl'
    received_size = 0
    with open(file_path, "wb") as f:
        while received_size < file_size:
            if received_size+chunk_size > file_size:
                chunk = conn.recv(file_size-received_size)
            else:
                chunk = conn.recv(chunk_size)
            if chunk:
                f.write(chunk)
                received_size += len(chunk)
    assert received_size == file_size
    print(f'Received {file_path} from client')
    compressed_data = load_compressed_data(file_path)
    with torch.no_grad():
        decoded_tensor = decoder(compressed_data)
    reconstructed_img = convert_tensor_to_image(
        decoded_tensor[0], YUV_ENABLED)
    reconstructed_path = base_path+'_rec'+IMAGE_EXT
    reconstructed_img.save(reconstructed_path)
    print(f'Decoding {file_path} to {reconstructed_path}')
    imgs = [reconstructed_path]
    gt_imgs = [base_path+IMAGE_EXT]
    if dnn_task == "yolov5":
        inputs = convert_image_to_tensor(imgs[0], False, True)
        inputs = [convert_tensor_to_numpy(inputs, False, False)]
        test_yolov5(inputs, gt_imgs, False)
    # Server CUDA and PyTorch versions are too low
    # else:
    #     inputs = convert_image_to_tensor(
    #         imgs[0], False, True).unsqueeze(0)
    #     gt_inputs = convert_image_to_tensor(
    #         gt_imgs[0], False, True).unsqueeze(0)
    #     if dnn_task == "faster_rcnn":
    #         test_faster_rcnn(imgs, inputs, gt_imgs, gt_inputs, False)
    #     elif dnn_task == "fcos":
    #         test_fcos(imgs, inputs, gt_imgs, gt_inputs, False)
    #     elif dnn_task == "ssd":
    #         test_ssd(imgs, inputs, gt_imgs, gt_inputs, False)
    print(f'Analyzed {reconstructed_path}')
    i += 1
conn.close()
server_socket.close()
