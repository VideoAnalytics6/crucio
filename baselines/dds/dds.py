import glob
import os
import shutil
import time

import torch
from networkx.algorithms.components.connected import connected_components

from autoencoder.dataset import (IMAGE_HEIGHT, IMAGE_WIDTH,
                                 load_video_to_tensor, VIDEO_DIR)
from baselines.dds.util import (Region, Results, compress_and_get_size,
                                crop_images, extract_images_from_video,
                                IMAGE_EXT, pairwise_overlap_indexing_list,
                                simple_merge, to_graph)
from dnn_model.faster_rcnn import load_faster_rcnn


def compute_regions_size(results, vid_name, images_direc, qp):
    vid_name = f"{vid_name}-cropped"
    start = time.time()
    frames_count = crop_images(results, vid_name, images_direc)
    size = compress_and_get_size(vid_name, 1, frames_count+1, qp=qp)
    end = time.time()
    encoding_time = end-start
    return size, encoding_time


def get_phase_results(dnn_task, dnn_batch_size, vid_name, req_regions, first=False):
    if first:
        images_direc = VIDEO_DIR+'/'+vid_name + "-base-phase-cropped"
    else:
        images_direc = VIDEO_DIR+'/'+vid_name + "-cropped"

    start = time.time()
    extract_images_from_video(images_direc, req_regions)
    end = time.time()
    decoding_time = end-start

    image_list = glob.glob(images_direc+'/*'+IMAGE_EXT)
    frame_num = len(image_list)
    iters = int(frame_num/dnn_batch_size)
    rem = frame_num-iters*dnn_batch_size
    if rem > 0:
        iters += 1
    t_results = []
    t_proposals = []
    t_scores = []
    id = 1
    start = time.time()
    for _ in range(iters):
        length = dnn_batch_size
        if rem > 0 and _ == iters-1:
            length = rem
        inputs, id = load_video_to_tensor(
            images_direc, start_num=id, length=length, return_list=True)
        with torch.no_grad():
            if first:
                _results, proposals, _scores = dnn_task(
                    inputs, RPN=True)
                t_proposals += proposals
                t_scores += _scores
            else:
                _results = dnn_task(inputs)
            t_results += _results
    end = time.time()
    infer_time = end-start

    n = 0
    i = 0
    fid = 1
    results = Results()
    if first:
        rpn = Results()
    scores_thresold = 0.99
    while n < frame_num:
        str_num = "{:04d}".format(fid)
        img_path = os.path.join(images_direc, 'frame'+str_num+IMAGE_EXT)
        if img_path in image_list:
            yy = len(t_results[i]['labels'])
            for j in range(yy):
                temp = t_results[i]['boxes'][j]
                x = temp[0]/IMAGE_WIDTH
                y = temp[1]/IMAGE_HEIGHT
                w = (temp[2]-temp[0])/IMAGE_WIDTH
                h = (temp[3]-temp[1])/IMAGE_HEIGHT
                conf = t_results[i]['scores'][j]
                label = t_results[i]['labels'][j]
                results.append(Region(fid, x, y, w, h, conf, label))
            if first:
                yy = len(t_proposals[i])
                for j in range(yy):
                    if t_scores[i][j] > scores_thresold:
                        temp = t_proposals[i][j]
                        x = temp[0]/IMAGE_WIDTH
                        y = temp[1]/IMAGE_HEIGHT
                        w = (temp[2]-temp[0])/IMAGE_WIDTH
                        h = (temp[3]-temp[1])/IMAGE_HEIGHT
                        rpn.append(Region(fid, x, y, w, h, t_scores[i][j], 2))
            n += 1
            i += 1
        fid += 1

    if first:
        return results, rpn, infer_time, decoding_time
    else:
        return results, infer_time, decoding_time


def cleanup(vid_name):
    if os.path.isdir(VIDEO_DIR+'/'+vid_name + "-cropped"):
        shutil.rmtree(VIDEO_DIR+'/'+vid_name + "-cropped")
    if os.path.isdir(VIDEO_DIR+'/'+vid_name + "-base-phase-cropped"):
        shutil.rmtree(VIDEO_DIR+'/'+vid_name + "-base-phase-cropped")


def merge_boxes_in_results(results_dict, min_conf_threshold, iou_threshold):
    final_results = Results()

    # Clean dict to remove min_conf_threshold
    for _, regions in results_dict.items():
        to_remove = []
        for r in regions:
            if r.conf < min_conf_threshold:
                to_remove.append(r)
        for r in to_remove:
            regions.remove(r)

    for fid, regions in results_dict.items():
        overlap_pairwise_list = pairwise_overlap_indexing_list(
            regions, iou_threshold)
        overlap_graph = to_graph(overlap_pairwise_list)
        grouped_bbox_idx = [c for c in sorted(
            connected_components(overlap_graph), key=len, reverse=True)]
        merged_regions = simple_merge(regions, grouped_bbox_idx)
        for r in merged_regions:
            final_results.append(r)
    return final_results


def dds_delay(dnn_task, dnn_batch_size, video_name):
    final_results = Results()
    all_required_regions = Results()
    low_phase_size = 0
    high_phase_size = 0
    low_infer_time = 0
    high_infer_time = 0
    low_encoding_time = 0
    high_encoding_time = 0
    low_decoding_time = 0
    high_decoding_time = 0
    low_qp = 60
    high_qp = 15
    intersection_threshold = 0.4

    file_type = '*'+IMAGE_EXT
    video_path = VIDEO_DIR+'/'+video_name
    file_list = glob.glob(os.path.join(video_path, file_type))
    frame_num = len(file_list)
    start_frame = 1
    end_frame = frame_num+1

    start = time.time()

    # First iteration
    req_regions = Results()
    for fid in range(start_frame, end_frame):
        req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2))
    low_phase_size, low_encoding_time = compute_regions_size(
        req_regions, f"{video_name}-base-phase", video_path, low_qp)
    results, rpn_regions, low_infer_time, low_decoding_time = get_phase_results(
        dnn_task, dnn_batch_size, video_name, req_regions, True)
    final_results.combine_results(results, intersection_threshold)
    all_required_regions.combine_results(rpn_regions, intersection_threshold)

    # Second Iteration
    if len(rpn_regions) > 0:
        high_phase_size, high_encoding_time = compute_regions_size(
            rpn_regions, video_name, video_path, high_qp)
        results, high_infer_time, high_decoding_time = get_phase_results(
            dnn_task, dnn_batch_size, video_name, rpn_regions)
        final_results.combine_results(results, intersection_threshold)

    cleanup(video_name)

    final_results = merge_boxes_in_results(
        final_results.regions_dict, 0.3, 0.3)
    final_results.fill_gaps(frame_num)
    final_results.combine_results(all_required_regions, intersection_threshold)
    final_results.write(f"{video_path}/dds_results.txt")

    end = time.time()
    total_time = end-start

    encoding_time = low_encoding_time+high_encoding_time
    data_size = (low_phase_size+high_phase_size)/1024
    decoding_time = low_decoding_time+high_decoding_time
    infer_time = low_infer_time+high_infer_time
    os.remove(f"{VIDEO_DIR}/{video_name}/dds_results.txt")
    return total_time, encoding_time, data_size, decoding_time, infer_time


if __name__ == '__main__':
    _, faster_rcnn_running = load_faster_rcnn()
    dds_delay(faster_rcnn_running, 8, 'cow_0004_015')
