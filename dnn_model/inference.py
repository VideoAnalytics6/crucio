import glob
import os

import torch

from autoencoder.dataset import IMAGE_DIR, IMAGE_EXT, load_video_to_tensor
from autoencoder.util import convert_image_to_tensor
from dnn_model.faster_rcnn import test_faster_rcnn
from dnn_model.fcos import test_fcos
from dnn_model.ssd import test_ssd
from dnn_model.util import convert_tensor_to_numpy
from dnn_model.yolov5 import test_yolov5


def image_inference():
    imgs = [IMAGE_DIR+'/000000013291_rec'+IMAGE_EXT]
    inputs = convert_image_to_tensor(imgs[0], False, True)
    inputs = [convert_tensor_to_numpy(inputs, False, False)]
    gt_imgs = [IMAGE_DIR+'/000000013291'+IMAGE_EXT]
    gt_inputs = convert_image_to_tensor(gt_imgs[0], False, True).unsqueeze(0)
    test_yolov5(inputs, gt_imgs)

    imgs = [IMAGE_DIR+'/000000013291_rec'+IMAGE_EXT]
    inputs = convert_image_to_tensor(imgs[0], False, True).unsqueeze(0)
    test_ssd(imgs, inputs, gt_imgs, gt_inputs)
    test_faster_rcnn(imgs, inputs, gt_imgs, gt_inputs)
    test_fcos(imgs, inputs, gt_imgs, gt_inputs)


def video_inference(dnn_task, video_path, dnn_batch_size=8):
    inference = []
    frame_num = len(glob.glob(os.path.join(video_path, '*'+IMAGE_EXT)))
    iters = int(frame_num/dnn_batch_size)
    rem = frame_num-iters*dnn_batch_size
    if rem > 0:
        iters += 1
    id = 1
    for _ in range(iters):
        length = dnn_batch_size
        if rem > 0 and _ == iters-1:
            length = rem
        inputs, id = load_video_to_tensor(
            video_path, start_num=id, length=length, return_list=True)
        with torch.no_grad():
            results = dnn_task(inputs)
            inference += results
    return inference


def crucio_tensor_inference(dnn_task, tensor, dnn_batch_size=8):
    tensor = tensor[0].permute(1, 0, 2, 3)
    frame_num = len(tensor)
    inference = []
    iters = int(frame_num/dnn_batch_size)
    rem = frame_num-iters*dnn_batch_size
    if rem > 0:
        iters += 1
    start = 0
    for _ in range(iters):
        length = dnn_batch_size
        if rem > 0 and _ == iters-1:
            length = rem
        end = start+length
        inputs = tensor[start:end, :, :, :]
        with torch.no_grad():
            results = dnn_task(inputs)
            inference += results
        start = end
    return inference


if __name__ == '__main__':
    image_inference()
