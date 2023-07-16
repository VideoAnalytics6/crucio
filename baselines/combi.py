import os
import shutil
import time

from autoencoder.dataset import VIDEO_DIR
from autoencoder.util import IMAGE_EXT
from baselines.dds.dds import dds_delay
from baselines.reducto.reducto import (EdgeDiff, differ, establish_thresh_map,
                                       feature)
from dnn_model.faster_rcnn import load_faster_rcnn


def combi_delay(thresh_map, dnn_task, dnn_batch_size, video_name, time_slot, frame_rate):
    video_path = VIDEO_DIR+'/'+video_name
    start = time.time()
    vector = differ.get_diff_vector(feature[0], video_path)
    diff_vectors = {feature[0]: vector}
    thresh, distance = thresh_map.get_thresh(diff_vectors[feature[0]])
    edge_diff = EdgeDiff(thresh=thresh)
    filter_results = edge_diff.filter_video(video_path)
    selected_frames = filter_results['selected_frames']
    # frame_num = int(time_slot*frame_rate)
    # selected_frames = random_selected_frames(frame_num)
    filter_name = video_name+'_filter'
    filter_path = VIDEO_DIR+'/'+filter_name
    os.mkdir(filter_path)
    curr_num = 1
    for i in selected_frames:
        src_path = f'{video_path}/frame{int(i):04d}{IMAGE_EXT}'
        dst_path = f'{filter_path}/frame{int(curr_num):04d}{IMAGE_EXT}'
        shutil.copy(src_path, dst_path)
        curr_num += 1
    end = time.time()
    filter_time = end-start

    total_time, encoding_time, data_size, decoding_time, infer_time = dds_delay(
        dnn_task, dnn_batch_size, filter_name)
    total_time += filter_time
    encoding_time += filter_time
    shutil.rmtree(filter_path)
    return total_time, encoding_time, data_size, decoding_time, infer_time


if __name__ == '__main__':
    _, faster_rcnn_running = load_faster_rcnn()
    thresh_map = establish_thresh_map(faster_rcnn_running)
    combi_delay(thresh_map, faster_rcnn_running,
                8, 'aeroplane_0001_001', 1.5, 30)
