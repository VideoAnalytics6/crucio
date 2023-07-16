import torch
from torchvision.models.detection.ssd import SSD300_VGG16_Weights, ssd300_vgg16

from dnn_model.util import f1_score, show_detections

ssd_running = None
ssd_weights = None


def load_ssd():
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights, score_thresh=0.5)
    model = model.cuda()
    model.eval()
    return weights, model


def test_ssd(imgs, inputs, gt_imgs, gt_inputs, show=True):
    global ssd_running, ssd_weights
    if ssd_running is None:
        ssd_weights, ssd_running = load_ssd()
    with torch.no_grad():
        results = ssd_running(inputs)
        gt_results = ssd_running(gt_inputs)

    if show:
        show_detections(ssd_weights, imgs, results, 0)
        show_detections(ssd_weights, gt_imgs, gt_results, 0)

    print(results[0])
    print(f1_score(results, gt_results)[0])
