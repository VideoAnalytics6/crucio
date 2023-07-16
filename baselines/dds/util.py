import os
import subprocess

import cv2
import networkx
import numpy as np

from autoencoder.dataset import VIDEO_DIR
from autoencoder.util import IMAGE_EXT


def calc_intersection_area(a, b):
    to = max(a.y, b.y)
    le = max(a.x, b.x)
    bo = min(a.y + a.h, b.y + b.h)
    ri = min(a.x + a.w, b.x + b.w)
    w = max(0, ri - le)
    h = max(0, bo - to)
    return w * h


def calc_area(a):
    w = max(0, a.w)
    h = max(0, a.h)
    return w * h


def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area
    return intersection_area / union_area


class Region:
    def __init__(self, fid, x, y, w, h, conf, label, origin="generic"):
        self.fid = int(fid)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.conf = float(conf)
        self.label = label
        self.origin = origin

    def is_same(self, region_to_check, threshold=0.5):
        # If fids or labels are different
        # then not same
        if (self.fid != region_to_check.fid or
                ((self.label != "-1" and region_to_check.label != "-1") and
                 (self.label != region_to_check.label))):
            return False

        # If intersection to union area
        # ratio is greater than threshold
        # then regions are same
        if calc_iou(self, region_to_check) > threshold:
            return True
        else:
            return False


class Results:
    def __init__(self):
        self.regions = []
        self.regions_dict = {}

    def __len__(self):
        return len(self.regions)

    def is_dup(self, result_to_add, threshold=0.5):
        # return regions with IOU greater than threshold
        # and maximum confidence
        if result_to_add.fid not in self.regions_dict:
            return None

        max_conf = -1
        max_conf_result = None
        for existing_result in self.regions_dict[result_to_add.fid]:
            if existing_result.is_same(result_to_add, threshold):
                if existing_result.conf > max_conf:
                    max_conf = existing_result.conf
                    max_conf_result = existing_result
        return max_conf_result

    def combine_results(self, additional_results, threshold=0.5):
        for result_to_add in additional_results.regions:
            self.add_single_result(result_to_add, threshold)

    def add_single_result(self, region_to_add, threshold=0.5):
        if threshold == 1:
            self.append(region_to_add)
            return
        dup_region = self.is_dup(region_to_add, threshold)
        if (not dup_region or
                ("tracking" in region_to_add.origin and
                 "tracking" in dup_region.origin)):
            self.regions.append(region_to_add)
            if region_to_add.fid not in self.regions_dict:
                self.regions_dict[region_to_add.fid] = []
            self.regions_dict[region_to_add.fid].append(region_to_add)
        else:
            final_object = None
            if dup_region.origin == region_to_add.origin:
                final_object = max([region_to_add, dup_region],
                                   key=lambda r: r.conf)
            elif ("low" in dup_region.origin and
                  "high" in region_to_add.origin):
                final_object = region_to_add
            elif ("high" in dup_region.origin and
                  "low" in region_to_add.origin):
                final_object = dup_region
            dup_region.x = final_object.x
            dup_region.y = final_object.y
            dup_region.w = final_object.w
            dup_region.h = final_object.h
            dup_region.conf = final_object.conf
            dup_region.origin = final_object.origin

    def append(self, region_to_add):
        self.regions.append(region_to_add)
        if region_to_add.fid not in self.regions_dict:
            self.regions_dict[region_to_add.fid] = []
        self.regions_dict[region_to_add.fid].append(region_to_add)

    def remove(self, region_to_remove):
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)
        self.regions.remove(region_to_remove)
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)

    def fill_gaps(self, frame_number):
        if len(self.regions) == 0:
            return
        results_to_add = Results()
        # max_resolution = max([e.resolution for e in self.regions])
        fids_in_results = [e.fid for e in self.regions]
        for i in range(frame_number):
            if i not in fids_in_results:
                results_to_add.regions.append(Region(i, 0, 0, 0, 0,
                                                     0.1, "no obj"))
        self.combine_results(results_to_add)
        self.regions.sort(key=lambda r: r.fid)

    def write(self, fname):
        results_file = open(fname, "w")
        for region in self.regions:
            # prepare string to write
            str_to_write = (f"{region.fid},{region.x},{region.y},"
                            f"{region.w},{region.h},"
                            f"{region.label},{region.conf},"
                            f"{region.origin}\n")
            results_file.write(str_to_write)
        results_file.close()


def crop_images(results, vid_name, images_direc):
    cached_image = None
    cropped_images = {}

    for region in results.regions:
        if not (cached_image and cached_image[0] == region.fid):
            image_path = os.path.join(
                images_direc, f"frame{str(region.fid).zfill(4)}{IMAGE_EXT}")
            cached_image = (region.fid, cv2.imread(image_path))

        # Just move complete image
        if region.x == 0 and region.y == 0 and region.w == 1 and region.h == 1:
            cropped_images[region.fid] = cached_image[1]
            continue

        width = cached_image[1].shape[1]
        height = cached_image[1].shape[0]
        x0 = int(region.x * width)
        y0 = int(region.y * height)
        x1 = int((region.w * width) + x0 - 1)
        y1 = int((region.h * height) + y0 - 1)

        if region.fid not in cropped_images:
            cropped_images[region.fid] = np.zeros_like(cached_image[1])

        cropped_image = cropped_images[region.fid]
        cropped_image[y0:y1, x0:x1, :] = cached_image[1][y0:y1, x0:x1, :]
        cropped_images[region.fid] = cropped_image

    os.makedirs(VIDEO_DIR+'/'+vid_name, exist_ok=True)
    frames_count = len(cropped_images)
    frames = sorted(cropped_images.items(), key=lambda e: e[0])

    for idx, (_, frame) in enumerate(frames):
        cv2.imwrite(os.path.join(VIDEO_DIR+'/'+vid_name,
                    f"frame{str(idx+1).zfill(4)}{IMAGE_EXT}"), frame)
    return frames_count


def compress_and_get_size(images_path, start_num, end_num, qp):
    images_path = VIDEO_DIR+'/'+images_path
    frame_number = end_num - start_num
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    encoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-loglevel", "error",
                                      "-start_number", str(start_num),
                                      '-i', f"{images_path}/frame%04d{IMAGE_EXT}",
                                      "-vcodec", "libx264",
                                      "-g", "15",
                                      "-keyint_min", "15",
                                      "-qp", f"{qp}",
                                      "-pix_fmt", "yuv420p",
                                      "-frames:v",
                                      str(frame_number),
                                      encoded_vid_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)

    size = 0
    if encoding_result.returncode != 0:
        # Encoding failed
        print("ENCODING FAILED")
        print(encoding_result.stdout)
        print(encoding_result.stderr)
        exit()
    else:
        size = os.path.getsize(encoded_vid_path)
    return size


def extract_images_from_video(images_path, req_regions):
    if not os.path.isdir(images_path):
        return

    for fname in os.listdir(images_path):
        if IMAGE_EXT[1:] not in fname:
            continue
        else:
            os.remove(os.path.join(images_path, fname))
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    extacted_images_path = os.path.join(images_path, "rec%04d"+IMAGE_EXT)
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",
                                      "-g", "8", "-q:v", "2",
                                      "-vsync", "0", "-start_number", "1",
                                      extacted_images_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()

    # os.remove(encoded_vid_path)

    fnames = sorted(
        [os.path.join(images_path, name)
         for name in os.listdir(images_path) if IMAGE_EXT[1:] in name])
    fids = sorted(list(set([r.fid for r in req_regions.regions])))
    fids_mapping = zip(fids, fnames)
    for fname in fnames:
        # Rename temporarily
        os.rename(fname, f"{fname}_temp")

    for fid, fname in fids_mapping:
        os.rename(os.path.join(f"{fname}_temp"), os.path.join(
            images_path, f"frame{str(fid).zfill(4)}{IMAGE_EXT}"))


def filter_bbox_group(bb1, bb2, iou_threshold):
    if calc_iou(bb1, bb2) > iou_threshold and bb1.label == bb2.label:
        return True
    else:
        return False


def pairwise_overlap_indexing_list(single_result_frame, iou_threshold):
    pointwise = [[i] for i in range(len(single_result_frame))]
    pairwise = [[i, j] for i, x in enumerate(single_result_frame)
                for j, y in enumerate(single_result_frame)
                if i != j if filter_bbox_group(x, y, iou_threshold)]
    return pointwise + pairwise


def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def simple_merge(single_result_frame, index_to_merge):
    # directly using largest box
    bbox_large = []
    for i in index_to_merge:
        i2np = np.array([j for j in i])
        left = min(np.array(single_result_frame)[i2np], key=lambda x: x.x)
        top = min(np.array(single_result_frame)[i2np], key=lambda x: x.y)
        right = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.x + x.w)
        bottom = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.y + x.h)

        fid, x, y, w, h, conf, label, origin = (
            left.fid, left.x, top.y, right.x + right.w - left.x,
            bottom.y + bottom.h - top.y, left.conf, left.label, left.origin)
        single_merged_region = Region(fid, x, y, w, h, conf,
                                      label, origin)
        bbox_large.append(single_merged_region)
    return bbox_large
