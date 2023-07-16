import json
import math
import os
import shutil
import time

import numpy as np
import torch

from autoencoder.dataset import (IMAGE_DIR, IMAGE_EXT, MIN_FRAME_NUM,
                                 VIDEO_DIR, load_video_to_tensor)
from autoencoder.network2d import get_networks
from autoencoder.network3d import get_networks3d
from autoencoder.util import (CRUCIO_DIR, GPU_ENABLED, YUV_ENABLED,
                              convert_image_to_tensor, load_compressed_data,
                              save_compressed_data)
from baselines.reducto.reducto import establish_thresh_map
from batch_handler.gru_filter import (apply_select_to_video, get_filter,
                                      scores_to_selects)
from dnn_model.faster_rcnn import load_faster_rcnn
from dnn_model.inference import crucio_tensor_inference

faster_rcnn_weights, faster_rcnn_running = load_faster_rcnn()
encoder, decoder = get_networks('eval', True)
img_path = IMAGE_DIR+'/000000013291'+IMAGE_EXT
img_tensor = convert_image_to_tensor(
    img_path, YUV_ENABLED, GPU_ENABLED).unsqueeze(0)
with torch.no_grad():
    compressed_data = encoder(img_tensor)
data_path = save_compressed_data(img_path, compressed_data)
compressed_data = load_compressed_data(data_path)
with torch.no_grad():
    decoded_tensor = decoder(compressed_data)
    results = faster_rcnn_running(decoded_tensor)
os.remove(data_path)

extractor, gru = get_filter('eval', True)
encoder3d, decoder3d = get_networks3d('eval', True)
thresh_map = establish_thresh_map(faster_rcnn_running)
with open(CRUCIO_DIR+'/research_work/conference_2/metric.json', 'r') as f:
    config = json.load(f)
dds_scale = config['dds_scale']
client_scale = config['client_scale']
server_scale = config['server_scale']
dnn_batch_size = config['dnn_batch_size']


def frame_num_changed_video(base_video_name, frame_num):
    base_video_path = VIDEO_DIR+'/'+base_video_name
    video_name = base_video_name+f'_n{frame_num}'
    video_path = VIDEO_DIR+'/'+video_name
    os.mkdir(video_path)
    id = 1
    for _ in range(frame_num):
        dst_num = "{:04d}".format(_+1)
        dst_path = os.path.join(video_path, 'frame'+dst_num+IMAGE_EXT)
        src_num = "{:04d}".format(id)
        src_path = os.path.join(base_video_path, 'frame'+src_num+IMAGE_EXT)
        if not os.path.exists(src_path):
            id = 1
            src_num = "{:04d}".format(id)
            src_path = os.path.join(base_video_path, 'frame'+src_num+IMAGE_EXT)
        shutil.copy(src_path, dst_path)
        id += 1
    return video_name, video_path


def adjust_batch_size(frame_num, batch_size):
    '''
    Adjust batch size according to MIN_FRAME_NUM
    '''
    if batch_size > frame_num:
        batch_size = frame_num
    batch_size = int(batch_size)
    if batch_size < MIN_FRAME_NUM:
        batch_size = MIN_FRAME_NUM
    rem_frame_num = frame_num % batch_size
    if 0 < rem_frame_num < MIN_FRAME_NUM:
        batch_frame_num = frame_num-MIN_FRAME_NUM
        if batch_size < batch_frame_num:
            large_size = batch_size
            while batch_frame_num % large_size != 0:
                large_size += 1
            small_size = batch_size
            while batch_frame_num % small_size != 0:
                if small_size == MIN_FRAME_NUM:
                    break
                small_size -= 1
            if small_size != MIN_FRAME_NUM:
                if large_size-batch_size < batch_size-small_size:
                    batch_size = large_size
                else:
                    batch_size = small_size
            else:
                batch_size = large_size
        else:
            batch_size = batch_frame_num
        print(f"Adjusted batch size is {batch_size}")
    return batch_size


def crucio_delay(video_name, frame_num, batch_size):
    video_path = VIDEO_DIR+'/'+video_name
    batch_size = adjust_batch_size(frame_num, batch_size)
    batch_num = math.ceil(frame_num/batch_size)

    filter_ratios = []
    encoding_delays = []
    data_sizes = []
    decoding_delays = []
    infer_delays = []
    start_num = 1
    for index in range(batch_num):
        data_size = 0
        start = time.time()
        for _ in range(index+1):
            video_tensor = load_video_to_tensor(
                video_path, start_num=start_num, length=batch_size).unsqueeze(0)
            with torch.no_grad():
                features, size = extractor(video_tensor)
                scores = gru(features)
        selects = scores_to_selects(scores)
        video_tensor = apply_select_to_video(selects, video_tensor)
        video_tensor = video_tensor[0].unsqueeze(0)
        for _ in range(index+1):
            with torch.no_grad():
                compressed_data = encoder3d(video_tensor)
            data_path = save_compressed_data(video_path, compressed_data)
            data_size += os.path.getsize(data_path)/1024
        end = time.time()
        encoding_delays.append(end-start)
        data_sizes.append(data_size)
        filter_ratios.append(1-video_tensor.shape[2]/batch_size)

        start = time.time()
        for _ in range(index+1):
            compressed_data = load_compressed_data(data_path)
            with torch.no_grad():
                decoded_tensor = decoder3d(compressed_data)
        end = time.time()
        decoding_delays.append(end-start)

        start = time.time()
        for _ in range(index+1):
            crucio_tensor_inference(faster_rcnn_running, decoded_tensor)
        end = time.time()
        infer_delays.append(end-start)

        start_num += batch_size
        os.remove(data_path)

    filter_ratio = np.mean(filter_ratios)
    encoding_time = np.mean(encoding_delays)
    data_size = np.mean(data_sizes)
    decoding_time = np.mean(decoding_delays)
    infer_time = np.mean(infer_delays)
    return filter_ratio, encoding_time, data_size, decoding_time, infer_time


crucio_delay('car_0008_015', 45, 21)
