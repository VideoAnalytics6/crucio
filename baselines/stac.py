import glob
import os
import shutil
import time

from autoencoder.dataset import VIDEO_DIR
from autoencoder.util import IMAGE_EXT
from baselines.reducto.reducto import (EdgeDiff, differ, establish_thresh_map,
                                       feature)
from dnn_model.faster_rcnn import load_faster_rcnn
from dnn_model.inference import video_inference


def stac_delay(thresh_map, dnn_task, dnn_batch_size, video_name):
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

    img_list = glob.glob(filter_path+'/*'+IMAGE_EXT)
    for _ in range(len(img_list)):
        img_list[_], extension = os.path.splitext(img_list[_])
    data_size = 0

    encoding_time = 0
    for img in img_list:
        img_path = img+IMAGE_EXT
        data_path = img+'.mp4'
        start = time.time()
        command = f'ffmpeg -i {img_path} -loglevel quiet -crf 30 -c:v libx264 {data_path}'
        os.system(command)
        end = time.time()
        os.remove(img_path)
        single_time = end-start
        if single_time > encoding_time:
            encoding_time = single_time
        data_size += os.path.getsize(data_path)/1024
    # Serial decoding (difficult to reach server simultaneously)
    start = time.time()
    for img in img_list:
        data_path = img+'.mp4'
        rec_img = img+IMAGE_EXT
        command = f'ffmpeg -i {data_path} -loglevel quiet -vsync 0 {rec_img}'
        os.system(command)
        os.remove(data_path)
    end = time.time()
    decoding_time = end-start
    start = time.time()
    video_inference(dnn_task, filter_path, dnn_batch_size)
    end = time.time()
    infer_time = end-start
    encoding_time += filter_time
    shutil.rmtree(filter_path)
    return encoding_time, data_size, decoding_time, infer_time


if __name__ == '__main__':
    _, faster_rcnn_running = load_faster_rcnn()
    thresh_map = establish_thresh_map(faster_rcnn_running)
    stac_delay(thresh_map, faster_rcnn_running, 8, 'aeroplane_0001_001')
