import torch
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2)

from dnn_model.util import f1_score, show_detections

faster_rcnn_running = None
faster_rcnn_weights = None


def load_faster_rcnn(index=1):
    score_thresh = 0.5
    if index == 1:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(
            weights=weights, box_score_thresh=score_thresh)
    elif index == 2:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=score_thresh)
    model = model.cuda()
    model.eval()
    return weights, model


def test_faster_rcnn(imgs, inputs, gt_imgs, gt_inputs, show=True):
    global faster_rcnn_running, faster_rcnn_weights
    if faster_rcnn_running is None:
        faster_rcnn_weights, faster_rcnn_running = load_faster_rcnn()
    with torch.no_grad():
        results = faster_rcnn_running(inputs)
        gt_results = faster_rcnn_running(gt_inputs)

    if show:
        show_detections(faster_rcnn_weights, imgs, results, 0)
        show_detections(faster_rcnn_weights, gt_imgs, gt_results, 0)

    print(results[0])
    print(f1_score(results, gt_results)[0])
