import math
import os

import torch
import torch.nn as nn

from autoencoder.dataset import (CNN_FRAME_NUM, IMAGE_HEIGHT, IMAGE_WIDTH,
                                 MIN_FRAME_NUM)
from autoencoder.util import DOWNLOAD_DIR, GPU_ENABLED

NETWORK_DEPTH_3D = 4
assert NETWORK_DEPTH_3D >= 2 and NETWORK_DEPTH_3D <= 4
FEATURE_CHANNEL_3D = 2
assert FEATURE_CHANNEL_3D == 1 or FEATURE_CHANNEL_3D == 2 or FEATURE_CHANNEL_3D == 4
BITS_CHANNEL_3D = 3*NETWORK_DEPTH_3D
assert BITS_CHANNEL_3D >= 2 and BITS_CHANNEL_3D <= 16
print('Network parameters of 3d autoencoder')
print(
    f'NETWORK_DEPTH_3D={NETWORK_DEPTH_3D} (Strong feature representation but longer training time)')
print(
    f'FEATURE_CHANNEL_3D={FEATURE_CHANNEL_3D} (Beneficial to feature extraction but consumes more GPU memory)')
print(
    f'BITS_CHANNEL_3D={BITS_CHANNEL_3D} (Retain more features but compressed data is larger)')
print(f'CNN_FRAME_NUM={CNN_FRAME_NUM}')

layer2_channels = BITS_CHANNEL_3D if NETWORK_DEPTH_3D == 2 else 64*FEATURE_CHANNEL_3D
layer3_channels = BITS_CHANNEL_3D if NETWORK_DEPTH_3D == 3 else 128*FEATURE_CHANNEL_3D

# Weight path for encoder and decoder
weights3d_dir = DOWNLOAD_DIR+'/weights_network3d_' + str(NETWORK_DEPTH_3D) + '.'+str(
    FEATURE_CHANNEL_3D) + '.'+str(BITS_CHANNEL_3D)+'.'+str(CNN_FRAME_NUM)+'_yuv'
if not os.path.exists(weights3d_dir):
    os.mkdir(weights3d_dir)
encoder3d_path = weights3d_dir+'/encoder_loss_3.pth'
decoder3d_path = weights3d_dir+'/decoder_loss_3.pth'
print(f'encoder3d_path={encoder3d_path}')
print(f'decoder3d_path={decoder3d_path}')


class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.conv_par = [
            [3, 32*FEATURE_CHANNEL_3D,
                (3, 3, 3), (1, 1, 1), (1, 1, 1)],
            [32*FEATURE_CHANNEL_3D, layer2_channels,
                (3, 3, 3), (1, 1, 1), (1, 1, 1)],
            [64*FEATURE_CHANNEL_3D, layer3_channels,
                (3, 3, 3), (1, 1, 1), (1, 1, 1)],
            [128*FEATURE_CHANNEL_3D, BITS_CHANNEL_3D,
             (3, 3, 3), (1, 1, 1), (1, 1, 1)]]
        self.max_par = [
            [(2, 2, 2), (1, 2, 2)],
            [(2, 2, 2), (1, 2, 2)],
            [(2, 2, 2), (1, 2, 2)],
            [(2, 2, 2), (1, 2, 2)]]
        self.conv1 = nn.Conv3d(self.conv_par[0][0], self.conv_par[0][1], kernel_size=self.conv_par[0]
                               [2], stride=self.conv_par[0][3], padding=self.conv_par[0][4])
        self.re1 = nn.ReLU(True)
        self.max1 = nn.MaxPool3d(
            kernel_size=self.max_par[0][0], stride=self.max_par[0][1])
        self.conv2 = nn.Conv3d(self.conv_par[1][0], self.conv_par[1][1], kernel_size=self.conv_par[1]
                               [2], stride=self.conv_par[1][3], padding=self.conv_par[1][4])
        self.re2 = nn.ReLU(True)
        self.max2 = nn.MaxPool3d(
            kernel_size=self.max_par[1][0], stride=self.max_par[1][1])
        if NETWORK_DEPTH_3D >= 3:
            self.conv3 = nn.Conv3d(self.conv_par[2][0], self.conv_par[2][1], kernel_size=self.conv_par[2]
                                   [2], stride=self.conv_par[2][3], padding=self.conv_par[2][4])
            self.re3 = nn.ReLU(True)
            self.max3 = nn.MaxPool3d(
                kernel_size=self.max_par[2][0], stride=self.max_par[2][1])
        if NETWORK_DEPTH_3D == 4:
            self.conv4 = nn.Conv3d(self.conv_par[3][0], self.conv_par[3][1], kernel_size=self.conv_par[3]
                                   [2], stride=self.conv_par[3][3], padding=self.conv_par[3][4])
            self.re4 = nn.ReLU(True)
            self.max4 = nn.MaxPool3d(
                kernel_size=self.max_par[3][0], stride=self.max_par[3][1])

    def forward(self, x):
        assert x.shape[2] >= MIN_FRAME_NUM
        x = self.conv1(x)
        x = self.re1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.re2(x)
        x = self.max2(x)
        if NETWORK_DEPTH_3D >= 3:
            x = self.conv3(x)
            x = self.re3(x)
            x = self.max3(x)
        if NETWORK_DEPTH_3D == 4:
            x = self.conv4(x)
            x = self.re4(x)
            x = self.max4(x)
        return x


class VideoDecoder(nn.Module):
    def __init__(self):
        super(VideoDecoder, self).__init__()
        self.convt_par = [
            [BITS_CHANNEL_3D, 128*FEATURE_CHANNEL_3D,
                (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)],
            [layer3_channels, 64*FEATURE_CHANNEL_3D,
                (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)],
            [layer2_channels, 32*FEATURE_CHANNEL_3D,
                (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)],
            [32*FEATURE_CHANNEL_3D, 3,
             (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)]]
        if NETWORK_DEPTH_3D == 4:
            self.convt1 = nn.ConvTranspose3d(self.convt_par[0][0], self.convt_par[0][1], kernel_size=self.convt_par[0]
                                             [2], stride=self.convt_par[0][3], padding=self.convt_par[0][4], output_padding=self.convt_par[0][5])
            self.re1 = nn.ReLU(True)
        if NETWORK_DEPTH_3D >= 3:
            self.convt2 = nn.ConvTranspose3d(self.convt_par[1][0], self.convt_par[1][1], kernel_size=self.convt_par[1]
                                             [2], stride=self.convt_par[1][3], padding=self.convt_par[1][4], output_padding=self.convt_par[1][5])
            self.re2 = nn.ReLU(True)
        self.convt3 = nn.ConvTranspose3d(self.convt_par[2][0], self.convt_par[2][1], kernel_size=self.convt_par[2]
                                         [2], stride=self.convt_par[2][3], padding=self.convt_par[2][4], output_padding=self.convt_par[2][5])
        self.re3 = nn.ReLU(True)
        self.convt4 = nn.ConvTranspose3d(self.convt_par[3][0], self.convt_par[3][1], kernel_size=self.convt_par[3]
                                         [2], stride=self.convt_par[3][3], padding=self.convt_par[3][4], output_padding=self.convt_par[3][5])
        self.re4 = nn.ReLU(True)

    def forward(self, x):
        if NETWORK_DEPTH_3D == 4:
            x = self.convt1(x)
            x = self.re1(x)
        if NETWORK_DEPTH_3D >= 3:
            x = self.convt2(x)
            x = self.re2(x)
        x = self.convt3(x)
        x = self.re3(x)
        x = self.convt4(x)
        x = self.re4(x)
        return x


def get_networks3d(mode='train', is_load=False):
    '''
    mode -> Set network mode to 'train' or 'eval'
    is_load -> Whether to load 3d encoder/decoder weights 
               for continued training or evaluation
    '''
    encoder = VideoEncoder()
    decoder = VideoDecoder()
    if mode == 'train':
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        encoder.train()
        decoder.train()
    elif mode == 'eval':
        if GPU_ENABLED:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
        encoder.eval()
        decoder.eval()
    if is_load:
        encoder.load_state_dict(torch.load(encoder3d_path))
        decoder.load_state_dict(torch.load(decoder3d_path))
    return encoder, decoder


def conv3d_output_shape(input_shape, output_channels, kernel_size, stride, padding):
    batch_size, input_channels, input_depth, input_height, input_width = input_shape
    output_depth = math.floor(
        (input_depth - kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
    output_height = math.floor(
        (input_height - kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
    output_width = math.floor(
        (input_width - kernel_size[2] + 2 * padding[2]) / stride[2]) + 1
    return (batch_size, output_channels, output_depth, output_height, output_width)


def maxpool3d_output_shape(input_shape, kernel_size, stride):
    batch_size, input_channels, input_depth, input_height, input_width = input_shape
    output_depth = math.floor((input_depth - kernel_size[0]) / stride[0]) + 1
    output_height = math.floor((input_height - kernel_size[1]) / stride[1]) + 1
    output_width = math.floor((input_width - kernel_size[2]) / stride[2]) + 1
    return (batch_size, input_channels, output_depth, output_height, output_width)


def conv_transpose3d_output_shape(input_shape, output_channels, kernel_size, stride, padding, output_padding):
    batch_size, input_channels, input_depth, input_height, input_width = input_shape
    output_depth = math.floor(
        (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0])
    output_height = math.floor(
        (input_height - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1])
    output_width = math.floor(
        (input_width - 1) * stride[2] - 2 * padding[2] + kernel_size[2] + output_padding[2])
    return (batch_size, output_channels, output_depth, output_height, output_width)


def show_shape_transform3d(total_layers, batch_size):
    conv_par = VideoEncoder().conv_par
    max_par = VideoEncoder().max_par
    convt_par = VideoDecoder().convt_par
    tensor_shape = (batch_size, 3, CNN_FRAME_NUM, IMAGE_HEIGHT, IMAGE_WIDTH)
    print(f'{tensor_shape}')
    for _ in range(NETWORK_DEPTH_3D):
        print(f'--------VideoEncoder.conv3d{_+1}--------')
        tensor_shape = conv3d_output_shape(
            tensor_shape, conv_par[_][1], conv_par[_][2], conv_par[_][3], conv_par[_][4])
        print(f'{tensor_shape}')
        print(f'--------VideoEncoder.max3d{_+1}--------')
        tensor_shape = maxpool3d_output_shape(
            tensor_shape, max_par[_][0], max_par[_][1])
        print(f'{tensor_shape}')
    index = total_layers-NETWORK_DEPTH_3D
    for _ in range(index, total_layers):
        print(f'--------VideoDecoder.convt3d{_+1}--------')
        tensor_shape = conv_transpose3d_output_shape(
            tensor_shape, convt_par[_][1], convt_par[_][2], convt_par[_][3], convt_par[_][4], convt_par[_][5])
        print(f'{tensor_shape}')


if __name__ == '__main__':
    show_shape_transform3d(4, 2)
