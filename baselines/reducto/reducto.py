import os
import random
import re
import shutil
import time
from pathlib import Path

from autoencoder.dataset import IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_DIR
from baselines.reducto.util import (DEFAULT_FPS, IMAGE_EXT, REDUCTO_CONFIG,
                                    SEGMENT_LEN, DiffComposer, EdgeDiff,
                                    HashBuilder, MetricComposer, ThreshMap)
from dnn_model.faster_rcnn import load_faster_rcnn
from dnn_model.inference import video_inference

feature = ["edge"]
differ = DiffComposer.from_jsonfile(
    REDUCTO_CONFIG+'/threshes/auburn.json', feature)
config = [{"type": "map", "class": "50"},
          {"type": "map", "class": "75"},
          {"type": "map", "class": "small"},
          {"type": "map", "class": "large"}]
evaluator = MetricComposer.from_json(config)


def img2video(frame_root, output_path, selected_frames=None, frame_pattern='frame????', extension=IMAGE_EXT[1:]):
    frame_root = Path(frame_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_frames is None:
        selected_frames = [int(re.findall('\d+', f.stem)[0]) for f in sorted(
            frame_root.iterdir()) if f.match(f'{frame_pattern}.{extension}')]
    frame_list = [
        f'{frame_root}/frame{int(i):04d}.{extension}' for i in selected_frames]
    frame_str = ' '.join(frame_list)
    command = f'cat {frame_str} | ' \
              f'ffmpeg -hide_banner -loglevel panic ' \
              f'-f image2pipe -i - {output_path} -crf 14'
    os.system(command)


def video2img(video_path, frame_root, extension=IMAGE_EXT[1:], scale=1):
    scale_str = f'{IMAGE_WIDTH // scale}:{IMAGE_HEIGHT // scale}'
    frame_root = Path(frame_root)
    frame_root.mkdir(parents=True, exist_ok=True)
    command = f'ffmpeg -hide_banner -loglevel quiet -r 1 -i {video_path} -r 1 -vf scale={scale_str} "{frame_root}/frame%04d.{extension}"'
    os.system(command)
    frames = [f for f in sorted(frame_root.iterdir())
              if f.match(f'frame????.{extension}')]
    os.remove(video_path)
    return len(frames)


def establish_thresh_map(dnn_task):
    diff_vectors_list = []
    evaluations_list = []
    chunk_num = 5
    curr_chunk_num = 0
    for subdir in os.listdir(VIDEO_DIR):
        video_path = os.path.join(VIDEO_DIR, subdir)
        if os.path.isdir(video_path):
            # -- inference -------------------------------------------------
            inference = video_inference(dnn_task, video_path)
            # -- diff ------------------------------------------------------
            vector = differ.get_diff_vector(feature[0], video_path)
            diff_vectors = {feature[0]: vector}
            diff_vectors_list.append(diff_vectors)
            diff_results = differ.process_video(video_path, diff_vectors)
            # -- evaluation ------------------------------------------------
            evaluations = evaluator.evaluate(inference, diff_results)
            evaluations_list.append(evaluations)
            curr_chunk_num += 1
            if curr_chunk_num == chunk_num:
                break

    threshmap_init_dict = HashBuilder().generate_threshmap(
        evaluations_list, diff_vectors_list)
    thresh_map = ThreshMap(threshmap_init_dict[feature[0]])
    return thresh_map


def random_selected_frames(frame_num):
    selected_frames = []
    for _ in range(frame_num):
        if bool(random.randint(0, 1)):
            selected_frames.append(_+1)
    print(selected_frames)
    return selected_frames


def reducto_delay(thresh_map, dnn_task, dnn_batch_size, video_name, time_slot, frame_rate):
    video_path = VIDEO_DIR+'/'+video_name
    start = time.time()
    vector = differ.get_diff_vector(feature[0], video_path)
    diff_vectors = {feature[0]: vector}
    thresh, distance = thresh_map.get_thresh(diff_vectors[feature[0]])
    edge_diff = EdgeDiff(thresh=thresh)
    filter_results = edge_diff.filter_video(video_path)
    selected_frames = filter_results['selected_frames']
    frame_num = int(time_slot*frame_rate)
    # selected_frames = random_selected_frames(frame_num)
    end = time.time()
    filter_ratio = 1-len(selected_frames)/frame_num
    encoding_time = end-start

    segment_num = max(1, int(frame_num/SEGMENT_LEN*frame_rate/DEFAULT_FPS))
    separate = len(selected_frames)//segment_num
    selected_frames_seg = []
    for i in range(segment_num):
        if i == segment_num-1:
            selected_frames_seg.append(selected_frames[i*separate:])
        else:
            selected_frames_seg.append(
                selected_frames[i*separate:(i+1)*separate])
    video_temp_path_seg = []

    start = time.time()
    for seg in range(segment_num):
        video_temp_path = video_path+f'/temp{seg+1}.mp4'
        img2video(video_path, video_temp_path, selected_frames_seg[seg])
        video_temp_path_seg.append(video_temp_path)
    end = time.time()
    encoding_time += end-start

    data_size = 0
    for seg in range(segment_num):
        data_size += os.path.getsize(video_temp_path_seg[seg])/1024
    start = time.time()
    for seg in range(segment_num):
        video2img(video_temp_path_seg[seg], video_path+f'_temp{seg+1}')
    end = time.time()
    decoding_time = end-start
    start = time.time()
    for seg in range(segment_num):
        video_inference(dnn_task, video_path+f'_temp{seg+1}', dnn_batch_size)
    end = time.time()
    infer_time = end-start

    for seg in range(segment_num):
        shutil.rmtree(video_path+f'_temp{seg+1}')
    return filter_ratio, encoding_time, data_size, decoding_time, infer_time


if __name__ == '__main__':
    _, faster_rcnn_running = load_faster_rcnn()
    thresh_map = establish_thresh_map(faster_rcnn_running)
    reducto_delay(thresh_map, faster_rcnn_running,
                  8, 'aeroplane_0001_001', 1.5, 30)
