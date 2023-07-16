import cv2
import numpy as np
import torch
from torchvision.io.image import read_image
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes


def convert_tensor_to_numpy(tensor, is_norm, is_yuv):
    '''
    Convert normalized tensor (an image) to NumPy array
    tensor -> Tensor output from decoder
    is_norm -> Whether to maintain normalized range
    is_yuv -> Whether tensor is in YUV color space
    '''
    if is_norm:
        array = tensor.detach().cpu().numpy()
    else:
        array = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    array = array.transpose((1, 2, 0))
    if is_yuv:
        array = cv2.cvtColor(array, cv2.COLOR_YUV2RGB)
    return array


def convert_numpy_to_tensor(array, is_gpu):
    '''
    Convert NumPy array to normalized tensor (an image) 
    array -> NumPy array
    is_gpu -> Whether tensor is moved to GPU
    '''
    array = array.transpose((2, 0, 1))
    tensor = torch.from_numpy(array)
    if is_gpu:
        tensor = tensor.cuda()
    return tensor


def show_detections(weights, imgs, results, index):
    '''
    Inference result is displayed as an image with a bounding box
    weights -> Model weights of analytics task
    imgs -> Input image array
    results -> Inference result array
    index -> Array index
    '''
    boxes = draw_bounding_boxes(
        read_image(imgs[index]),
        boxes=results[index]["boxes"],
        labels=[weights.meta["categories"][i]
                for i in results[index]["labels"]],
        width=4)
    img = to_pil_image(boxes)
    img.show()


def get_boxes_and_labels(results, index, is_yolov5):
    '''
    Returns a bounding box and label for inference result
    results -> Inference result array
    index -> Array index
    is_yolov5 -> Whether analytics task is YOLOv5
    '''
    if is_yolov5:
        boxes = torch.tensor(
            results.pandas().xyxy[index].values[:, 0:4].astype(np.float64))
        labels = torch.tensor(
            results.pandas().xyxy[index].values[:, 5].astype(np.int32))
    else:
        boxes = results[index]["boxes"]
        labels = results[index]["labels"]
    return boxes, labels


def f1_score(results, gt_results, is_yolov5=False, min_iou=0.7):
    '''
    Calculating F1 score [0,1] of current results based on Ground Truth 
    results -> Inference result array
    gt_results -> Inference result array (Ground Truth)
    is_yolov5 -> Whether analytics task is YOLOv5
    min_iou -> Minimum IOU requirements for bounding box overlap
    '''
    assert len(results) == len(gt_results)
    number = len(gt_results)
    f1 = []
    for i in range(number):
        boxes, labels = get_boxes_and_labels(results, i, is_yolov5)
        tp_and_fp = boxes.shape[0]
        gt_boxes, gt_labels = get_boxes_and_labels(gt_results, i, is_yolov5)
        tp_and_fn = gt_boxes.shape[0]
        score = 0
        if tp_and_fn == 0:
            score = 1
        else:
            if tp_and_fp != 0:
                iou = box_iou(boxes, gt_boxes)
                best_match = torch.nonzero(iou >= min_iou).tolist()
                tp = 0
                for j in range(len(best_match)):
                    index = best_match[j][0]
                    gt_index = best_match[j][1]
                    if labels[index] == gt_labels[gt_index]:
                        tp += 1
                # tp/(tp+fp)
                precision = tp/tp_and_fp
                # tp/(tp+fn)
                recall = tp/tp_and_fn
                if tp != 0:
                    score = 2/(1/precision+1/recall)
        f1.append(score)
    return f1
