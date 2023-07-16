import os

import torch

from autoencoder.util import home
from dnn_model.util import f1_score

yolov5_running = None


def load_yolov5(weights=home+'/.cache/torch/hub/checkpoints/yolov5x.pt'):
    # rm -rf ~/.cache/torch/hub/ultralytics_yolov5_master
    yolov5 = home+'/.cache/torch/hub/ultralytics_yolov5_master'
    score_thresh = 0.5
    if os.path.isdir(yolov5):
        model = torch.hub.load(yolov5, 'custom', weights, source='local')
    else:
        weights = os.path.splitext(os.path.basename(weights))[0]
        model = torch.hub.load('ultralytics/yolov5',
                               weights, trust_repo=True, pretrained=True)
    model.conf = score_thresh
    model = model.cuda()
    model.eval()
    return model


def test_yolov5(inputs, gt_inputs, show=True):
    global yolov5_running
    if yolov5_running is None:
        yolov5_running = load_yolov5()
    with torch.no_grad():
        results = yolov5_running(inputs)
        gt_results = yolov5_running(gt_inputs)

    if show:
        results.show()
        gt_results.show()

    print(results.pandas().xyxy[0])
    print(f1_score(results, gt_results, True)[0])
