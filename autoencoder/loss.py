import pytorch_msssim
import torch.nn as nn

from dnn_model.faster_rcnn import load_faster_rcnn
from dnn_model.fcos import load_fcos
from dnn_model.ssd import load_ssd


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, outputs, imgs):
        # Image is automatically normalized when to tensor, so data range here is 1
        return 1 - pytorch_msssim.ssim(outputs, imgs, data_range=1.0, size_average=True)


class FCOSLoss(nn.Module):
    def __init__(self):
        super(FCOSLoss, self).__init__()
        _, self.model = load_fcos()

    def forward(self, outputs, imgs):
        self.model.eval()
        ground_truths = self.model(imgs)
        self.model.train()
        loss = self.model(images=outputs, targets=ground_truths)
        bbox_regression = loss["bbox_regression"]
        bbox_ctrness = loss["bbox_ctrness"]
        classification = loss["classification"]
        return bbox_regression+bbox_ctrness+classification


class FasterRCNNLoss(nn.Module):
    def __init__(self):
        super(FasterRCNNLoss, self).__init__()
        _, self.model = load_faster_rcnn()

    def forward(self, outputs, imgs):
        self.model.eval()
        ground_truths = self.model(imgs)
        self.model.train()
        loss = self.model(images=outputs, targets=ground_truths)
        loss_box_reg = loss["loss_box_reg"]
        loss_classifier = loss["loss_classifier"]
        loss_objectness = loss["loss_objectness"]
        loss_rpn_box_reg = loss["loss_rpn_box_reg"]
        return loss_box_reg+loss_classifier+loss_objectness+loss_rpn_box_reg


class SSDLoss(nn.Module):
    def __init__(self):
        super(SSDLoss, self).__init__()
        _, self.model = load_ssd()

    def forward(self, outputs, imgs):
        self.model.eval()
        ground_truths = self.model(imgs)
        self.model.train()
        loss = self.model(images=outputs, targets=ground_truths)
        bbox_regression = loss["bbox_regression"]
        classification = loss["classification"]
        return bbox_regression+classification


def loss_function(index):
    '''
    Returns loss function based on index
    Note: Weights trained by loss function 3 must be loaded
          before using loss functions 4, 5, and 6
    1 -> L1Loss
    2 -> MSELoss或PSNRLoss
    3 -> SSIMLoss
    4 -> FCOSLoss
    5 -> FasterRCNNLoss
    6 -> SSDLoss
    '''
    if index == 1:
        print('Using L1Loss')
        return nn.L1Loss()
    elif index == 2:
        print('Using MSELoss')
        return nn.MSELoss()
    elif index == 3:
        print('Using SSIMLoss')
        return SSIMLoss()
    elif index == 4:
        print('Using FCOSLoss')
        return FCOSLoss()
    elif index == 5:
        print('Using FasterRCNNLoss')
        return FasterRCNNLoss()
    elif index == 6:
        print('Using SSDLoss')
        return SSDLoss()
