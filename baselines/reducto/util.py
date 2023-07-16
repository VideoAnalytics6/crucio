import configparser
import copy
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from sklearn import neighbors
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from autoencoder.util import CRUCIO_DIR, IMAGE_EXT

SEGMENT_LEN = 20
DEFAULT_FPS = 25
REDUCTO_CONFIG = CRUCIO_DIR+"/baselines/reducto/config"


def load_json(path):
    if path is None or not Path(path).exists():
        return None
    with open(path, 'r') as j:
        data = json.load(j)
    return data


def flatten(lists):
    return [item for sublist in lists for item in sublist]


class Metrics:
    def __init__(self, target_classes=None):
        self.target_classes = target_classes

    @staticmethod
    def interp_frame_ids(frame_ids, num_frames):
        full_ids, interped_ids = list(range(1, num_frames + 1)), []
        full_index, interp_index, last_number_seen = 0, 0, 0
        while full_index != len(full_ids) and interp_index != len(frame_ids):
            if full_ids[full_index] == frame_ids[interp_index]:
                last_number_seen = frame_ids[interp_index]
                interped_ids.append(last_number_seen)
                full_index += 1
                interp_index += 1
            else:
                interped_ids.append(last_number_seen)
                full_index += 1
        while full_index != len(full_ids):
            interped_ids.append(last_number_seen)
            full_index += 1
        return interped_ids

    @staticmethod
    def str2class(name):
        return {
            'map': MapMetrics,
        }[name]


class MapMetrics(Metrics):
    def __init__(self, target_classes=None):
        super(MapMetrics, self).__init__(target_classes)
        self.identifier = 'map'
        self.name = f'{self.identifier}_{self.target_classes}'


class MetricComposer:
    def __init__(self, metric_list=None):
        self.metric_list = metric_list
        self.keys = [metric.name for metric in metric_list]

    @staticmethod
    def from_json(config):
        return MetricComposer([
            Metrics.str2class(c['type'])(c['class'])
            for c in config
        ])

    def evaluate_single_frame(self, ground_truth, comparision):
        results = {}
        ground_truth = copy.deepcopy(ground_truth)
        del ground_truth['scores']
        metric = MeanAveragePrecision()
        metric.update([comparision], [ground_truth])
        results = metric.compute()
        return results

    @staticmethod
    def get_frame_pairs(inference, diff_results):
        selected_frame_list = flatten([
            [result['selected_frames']
                for _, result in diff_results[key]['result'].items()]
            for key in diff_results.keys()
        ])
        num_frames = len(inference)
        per_frame_list = set(flatten([
            list(zip(list(range(num_frames)),
                 Metrics.interp_frame_ids(frames, num_frames)))
            for frames in selected_frame_list
        ]))
        return per_frame_list

    def evaluate_per_frame(self, inference, diff_results):
        per_frame_list = self.get_frame_pairs(inference, diff_results)
        per_frame_eval = {
            frame_pair: self.evaluate_single_frame(
                inference[frame_pair[0]], inference[frame_pair[1]])
            for frame_pair in per_frame_list
        }
        return per_frame_eval

    def evaluate(self, inference, diff_results, per_frame_eval=None):
        per_frame_eval = per_frame_eval or self.evaluate_per_frame(
            inference, diff_results)
        evaluations = {}
        for differ_type, thresh_result in diff_results.items():
            evaluations[differ_type] = {}
            for thresh, diff_result in thresh_result['result'].items():
                selected_frames = diff_result['selected_frames']
                num_frames = len(inference)
                frame_pairs = zip(list(range(num_frames)), Metrics.interp_frame_ids(
                    selected_frames, num_frames))
                diff_thresh_evaluation = [per_frame_eval[fp]
                                          for fp in frame_pairs]
                evaluation = {}
                for key in self.keys:
                    evals = [abs(dte[key]) for dte in diff_thresh_evaluation]
                    evaluation[key] = sum(evals) / len(evals)
                evaluations[differ_type][thresh] = evaluation
        return evaluations


class DiffComposer:
    def __init__(self, differ_dict=None):
        self.differ_dict = differ_dict

    @ staticmethod
    def from_jsonfile(jsonpath, differencer_types=None):
        differencer_types = differencer_types or [
            'pixel', 'area', 'corner', 'edge']
        differ_dict = load_json(jsonpath)
        differencers = {
            feature: threshes
            for feature, threshes in differ_dict.items()
            if feature in differencer_types
        }
        return DiffComposer(differencers)

    def process_video(self, filepath, diff_vectors=None):
        if diff_vectors:
            assert all([k in diff_vectors for k in self.differ_dict.keys()])
            'not compatible diff-vector list'
        else:
            diff_vectors = {
                k: self.get_diff_vector(k, filepath)
                for k in self.differ_dict.keys()
            }
        results = {}
        for differ_type, thresholds in self.differ_dict.items():
            diff_vector = diff_vectors[differ_type]
            result = self.batch_diff(diff_vector, thresholds)
            results[differ_type] = {
                'diff_vector': diff_vector,
                'result': result,
            }
        return results

    @ staticmethod
    def get_diff_vector(differ_type, video_path):
        differ = DiffProcessor.str2class(differ_type)()
        return differ.get_diff_vector(video_path)

    @ staticmethod
    def batch_diff(diff_vector, thresholds):
        result = DiffProcessor.batch_diff_noobj(diff_vector, thresholds)
        return result


class DiffProcessor:

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        """
        :param thresh: threshold, frame with diff above which will be sent
        :param fraction: only support first and second, force fraction
        :param dataset: for loading external config
        """
        self.feature = 'none'
        self.fraction = fraction
        self.thresh = thresh
        self.section = self.get_section(dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'

    def get_diff_vector(self, video_path):
        diff_values = []
        frame_num = len(glob.glob(os.path.join(video_path, '*'+IMAGE_EXT)))
        img_path = os.path.join(video_path, 'frame0001'+IMAGE_EXT)
        prev_frame = cv2.imread(img_path)
        prev_frame = self.get_frame_feature(prev_frame)
        for _ in range(2, frame_num+1, 1):
            _ = "{:04d}".format(_)
            img_path = os.path.join(video_path, 'frame'+_+IMAGE_EXT)
            frame = cv2.imread(img_path)
            frame = self.get_frame_feature(frame)
            diff_value = self.cal_frame_diff(frame, prev_frame)
            diff_values.append(diff_value)
            prev_frame = frame
        return diff_values

    def filter_video(self, video_path):
        selected_frames = [1]
        estimations = [1.0]
        total_frames = len(glob.glob(os.path.join(video_path, '*'+IMAGE_EXT)))
        img_path = os.path.join(video_path, 'frame0001'+IMAGE_EXT)
        prev_frame = cv2.imread(img_path)
        prev_feat = self.get_frame_feature(prev_frame)
        for i in range(2, total_frames+1, 1):
            _ = "{:04d}".format(i)
            img_path = os.path.join(video_path, 'frame'+_+IMAGE_EXT)
            frame = cv2.imread(img_path)
            feat = self.get_frame_feature(frame)
            dis = self.cal_frame_diff(feat, prev_feat)
            if dis > self.thresh:
                selected_frames.append(i)
                prev_feat = feat
                estimations.append(1.0)
            else:
                estimations.append((self.thresh - dis) / self.thresh)
        result = {
            'feature': self.feature,
            'thresh': self.thresh,
            'selected_frames': selected_frames,
            'num_selected_frames': len(selected_frames),
            'num_total_frames': total_frames,
            'fraction': len(selected_frames) / total_frames,
            'estimation': sum(estimations) / len(estimations)
        }
        return result

    @ staticmethod
    def batch_diff_noobj(diff_value, thresholds):
        diff_integral = np.cumsum([0.0] + diff_value).tolist()
        diff_results = {}
        total_frames = 1 + len(diff_value)
        for thresh in thresholds:
            selected_frames = [1]
            estimations = [1.0]
            last, current = 1, 2
            while current < total_frames:
                diff_delta = diff_integral[current] - diff_integral[last]
                if diff_delta >= thresh:
                    selected_frames.append(current)
                    last = current
                    estimations.append(1.0)
                else:
                    estimations.append((thresh - diff_delta) / thresh)
                current += 1
            diff_results[thresh] = DiffProcessor._format_result(
                selected_frames, total_frames, estimations)
        return diff_results

    def cal_frame_diff(self, frame, prev_frame):
        """Calculate different between frames."""
        raise NotImplementedError()

    def get_frame_feature(self, frame):
        """Extract feature of frame."""
        raise NotImplementedError()

    @ staticmethod
    def get_section(dataset):
        config = configparser.ConfigParser()
        config.read(REDUCTO_CONFIG+'/diff_config.ini')
        return config[dataset if dataset and dataset in config else 'default']

    def _load_section(self, section):
        return

    def __str__(self):
        return self.name

    @ staticmethod
    def _format_result(selected_frames, total_frames, estimations):
        return {
            # 'fps': total_frames / complete_time if complete_time != 0 else -1,
            'selected_frames': selected_frames,
            'num_selected_frames': len(selected_frames),
            'num_total_frames': total_frames,
            'fraction': len(selected_frames) / total_frames,
            'estimation': sum(estimations) / len(estimations)
        }

    @ staticmethod
    def str2class(feature):
        return {
            'edge': EdgeDiff,
        }[feature]


class EdgeDiff(DiffProcessor):
    feature = 'edge'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.edge_blur_rad, self.edge_blur_rad),
                                self.edge_blur_var)
        edge = cv2.Canny(blur, self.edge_canny_low, self.edge_canny_high)
        return edge

    def cal_frame_diff(self, edge, prev_edge):
        total_pixels = edge.shape[0] * edge.shape[1]
        frame_diff = cv2.absdiff(edge, prev_edge)
        frame_diff = cv2.threshold(frame_diff, self.edge_thresh_low_bound, 255,
                                   cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

    def _load_section(self, section):
        self.edge_blur_rad = section.getint('EDGE_BLUR_RAD', 11)
        self.edge_blur_var = section.getint('EDGE_BLUR_VAR', 0)
        self.edge_canny_low = section.getint('EDGE_CANNY_LOW', 101)
        self.edge_canny_high = section.getint('EDGE_CANNY_HIGH', 255)
        self.edge_thresh_low_bound = section.getint(
            'EDGE_THRESH_LOW_BOUND', 21)


class HashBuilder:
    def __init__(self):
        # hyper-parameter
        self.feature_dim = 30
        self.knn_neighbors = 5
        # NOTE n_neighbors must be greater than number of chunk
        self.knn_weights = 'distance'

    def _histogram(self, diff_vector, distri_range):
        hist, _ = np.histogram(
            diff_vector, self.feature_dim, range=distri_range)
        return hist / len(diff_vector)

    @ staticmethod
    def _get_optimal_thresh(er, target_acc):
        optimal_thresh = 0.0
        for thresh, result in er.items():
            thresh = float(thresh)
            result_cross_query = min([abs(x) for x in result.values()])
            if result_cross_query > target_acc and thresh > optimal_thresh:
                optimal_thresh = thresh
        return optimal_thresh

    def generate_threshmap(self, evaluation_results_list, diff_vectors_list, target_acc=0.90, safe_zone=0.025):
        diff_value_range = {}
        for dp_dv in diff_vectors_list:
            for dp, dv in dp_dv.items():
                if dp not in diff_value_range:
                    diff_value_range[dp] = (min(dv), max(dv))
                else:
                    diff_value_range[dp] = (
                        min([min(dv), diff_value_range[dp][0]]),
                        max([max(dv), diff_value_range[dp][1]])
                    )
        optimal_thresh = {}
        thresh_candidate = {}
        for dp_er, dp_dv in zip(evaluation_results_list, diff_vectors_list):
            for dp, er in dp_er.items():
                dv = dp_dv[dp]
                if dp not in optimal_thresh:
                    optimal_thresh[dp] = []
                if dp not in thresh_candidate:
                    thresh_candidate[dp] = list(er.keys())
                optimal_thresh[dp].append((
                    self._histogram(dv, diff_value_range[dp]),
                    self._get_optimal_thresh(er, target_acc + safe_zone)
                ))
        hash_table = {}
        for dp, chunk_state in optimal_thresh.items():
            knn = neighbors.KNeighborsClassifier(
                self.knn_neighbors, weights=self.knn_weights)
            x = np.array([x[0] for x in chunk_state])
            _y = [(thresh_candidate[dp].index(opt) if opt in thresh_candidate[dp] else 0)
                  for opt in [x[1] for x in chunk_state]]
            y = np.array(_y)
            knn.fit(x, y)
            hash_table[dp] = {
                'table': knn,
                'distri': diff_value_range[dp],
                'dim': self.feature_dim,
                'tcand': thresh_candidate[dp]
            }
        return hash_table


class ThreshMap:
    def __init__(self, init_dict):
        self.hash_table = init_dict['table']
        self.distri_range = init_dict['distri']
        self.feature_dim = init_dict['dim']
        self.thresh_candidate = init_dict['tcand']

    def _histogram(self, diff_vector):
        hist, _ = np.histogram(
            diff_vector, self.feature_dim, range=self.distri_range)
        return hist / len(diff_vector)

    def get_thresh(self, diff_vector):
        diff_vector = np.array(self._histogram(diff_vector))[np.newaxis, :]
        pred_thresh = self.hash_table.predict(diff_vector).item()
        distance, _ = self.hash_table.kneighbors(
            diff_vector, return_distance=True)
        return self.thresh_candidate[pred_thresh], distance
