import math
import os

import torch
import torch.nn as nn

from autoencoder.dataset import IMAGE_HEIGHT, IMAGE_WIDTH
from autoencoder.util import DOWNLOAD_DIR, GPU_ENABLED

NETWORK_DEPTH = 4
assert NETWORK_DEPTH >= 2 and NETWORK_DEPTH <= 4
FEATURE_CHANNEL = 4
assert FEATURE_CHANNEL == 1 or FEATURE_CHANNEL == 2 or FEATURE_CHANNEL == 4
BITS_CHANNEL = 3*NETWORK_DEPTH
assert BITS_CHANNEL >= 2 and BITS_CHANNEL <= 16
print('Network parameters of autoencoder')
print(
    f'NETWORK_DEPTH={NETWORK_DEPTH} (Strong feature representation but longer training time)')
print(
    f'FEATURE_CHANNEL={FEATURE_CHANNEL} (Beneficial to feature extraction but consumes more GPU memory)')
print(
    f'BITS_CHANNEL={BITS_CHANNEL} (Retain more features but compressed data is larger)')

layer2_channels = BITS_CHANNEL if NETWORK_DEPTH == 2 else 64*FEATURE_CHANNEL
layer3_channels = BITS_CHANNEL if NETWORK_DEPTH == 3 else 128*FEATURE_CHANNEL

# Weight path for encoder and decoder
weights_dir = DOWNLOAD_DIR+'/weights_network_' + \
    str(NETWORK_DEPTH)+'.'+str(FEATURE_CHANNEL) + \
    '.'+str(BITS_CHANNEL)+'_yuv'
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)
encoder_path = weights_dir+'/encoder_loss_3.pth'
decoder_path = weights_dir+'/decoder_loss_3.pth'
print(f'encoder_path={encoder_path}')
print(f'decoder_path={decoder_path}')


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_par = [
            [3, 32*FEATURE_CHANNEL, 3, 1, 1],
            [32*FEATURE_CHANNEL, layer2_channels, 3, 1, 1],
            [64*FEATURE_CHANNEL, layer3_channels, 3, 1, 1],
            [128*FEATURE_CHANNEL, BITS_CHANNEL, 3, 1, 1]]
        self.max_par = [
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2]]
        self.conv1 = nn.Conv2d(self.conv_par[0][0], self.conv_par[0][1], kernel_size=self.conv_par[0]
                               [2], stride=self.conv_par[0][3], padding=self.conv_par[0][4])
        self.max1 = nn.MaxPool2d(
            kernel_size=self.max_par[0][0], stride=self.max_par[0][1])
        self.re1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.conv_par[1][0], self.conv_par[1][1], kernel_size=self.conv_par[1]
                               [2], stride=self.conv_par[1][3], padding=self.conv_par[1][4])
        self.max2 = nn.MaxPool2d(
            kernel_size=self.max_par[1][0], stride=self.max_par[1][1])
        if NETWORK_DEPTH >= 3:
            self.re2 = nn.ReLU()
            self.conv3 = nn.Conv2d(self.conv_par[2][0], self.conv_par[2][1], kernel_size=self.conv_par[2]
                                   [2], stride=self.conv_par[2][3], padding=self.conv_par[2][4])
            self.max3 = nn.MaxPool2d(
                kernel_size=self.max_par[2][0], stride=self.max_par[2][1])
        if NETWORK_DEPTH == 4:
            self.re3 = nn.ReLU()
            self.conv4 = nn.Conv2d(self.conv_par[3][0], self.conv_par[3][1], kernel_size=self.conv_par[3]
                                   [2], stride=self.conv_par[3][3], padding=self.conv_par[3][4])
            self.max4 = nn.MaxPool2d(
                kernel_size=self.max_par[3][0], stride=self.max_par[3][1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.re1(x)
        x = self.conv2(x)
        x = self.max2(x)
        if NETWORK_DEPTH >= 3:
            x = self.re2(x)
            x = self.conv3(x)
            x = self.max3(x)
        if NETWORK_DEPTH == 4:
            x = self.re3(x)
            x = self.conv4(x)
            x = self.max4(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convt_par = [
            [BITS_CHANNEL, 128*FEATURE_CHANNEL, 3, 2, 1, 1],
            [layer3_channels, 64*FEATURE_CHANNEL, 3, 2, 1, 1],
            [layer2_channels, 32*FEATURE_CHANNEL, 3, 2, 1, 1],
            [32*FEATURE_CHANNEL, 3, 3, 2, 1, 1]]
        if NETWORK_DEPTH == 4:
            self.convt1 = nn.ConvTranspose2d(self.convt_par[0][0], self.convt_par[0][1], kernel_size=self.convt_par[0]
                                             [2], stride=self.convt_par[0][3], padding=self.convt_par[0][4], output_padding=self.convt_par[0][5])
            self.re1 = nn.ReLU()
        if NETWORK_DEPTH >= 3:
            self.convt2 = nn.ConvTranspose2d(self.convt_par[1][0], self.convt_par[1][1], kernel_size=self.convt_par[1]
                                             [2], stride=self.convt_par[1][3], padding=self.convt_par[1][4], output_padding=self.convt_par[1][5])
            self.re2 = nn.ReLU()
        self.convt3 = nn.ConvTranspose2d(self.convt_par[2][0], self.convt_par[2][1], kernel_size=self.convt_par[2]
                                         [2], stride=self.convt_par[2][3], padding=self.convt_par[2][4], output_padding=self.convt_par[2][5])
        self.re3 = nn.ReLU()
        self.convt4 = nn.ConvTranspose2d(self.convt_par[3][0], self.convt_par[3][1], kernel_size=self.convt_par[3]
                                         [2], stride=self.convt_par[3][3], padding=self.convt_par[3][4], output_padding=self.convt_par[3][5])
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        if NETWORK_DEPTH == 4:
            x = self.convt1(x)
            x = self.re1(x)
        if NETWORK_DEPTH >= 3:
            x = self.convt2(x)
            x = self.re2(x)
        x = self.convt3(x)
        x = self.re3(x)
        x = self.convt4(x)
        x = self.sigm(x)
        return x


def get_networks(mode='train', is_load=False):
    '''
    mode -> Set network mode to 'train' or 'eval'
    is_load -> Whether to load encoder/decoder weights 
               for continued training or evaluation
    '''
    encoder = Encoder()
    decoder = Decoder()
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
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))
    return encoder, decoder


def conv_output_shape(input_shape, output_channels, kernel_size, stride, padding):
    batch_size, input_channels, input_height, input_width = input_shape
    output_height = math.floor(
        (input_height - kernel_size + 2*padding) / stride) + 1
    output_width = math.floor(
        (input_width - kernel_size + 2*padding) / stride) + 1
    return (batch_size, output_channels, output_height, output_width)


def maxpool_output_shape(input_shape, kernel_size, stride):
    batch_size, input_channels, input_height, input_width = input_shape
    output_height = math.floor((input_height - kernel_size) / stride) + 1
    output_width = math.floor((input_width - kernel_size) / stride) + 1
    return (batch_size, input_channels, output_height, output_width)


def conv_transpose_output_shape(input_shape, output_channels, kernel_size, stride, padding, output_padding):
    batch_size, input_channels, input_height, input_width = input_shape
    output_height = math.floor(
        (input_height - 1) * stride - 2 * padding + kernel_size + output_padding)
    output_width = math.floor(
        (input_width - 1) * stride - 2 * padding + kernel_size + output_padding)
    return (batch_size, output_channels, output_height, output_width)


def show_shape_transform(total_layers, batch_size):
    conv_par = Encoder().conv_par
    max_par = Encoder().max_par
    convt_par = Decoder().convt_par
    tensor_shape = (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    print(f'{tensor_shape}')
    for _ in range(NETWORK_DEPTH):
        print(f'--------Encoder.conv{_+1}--------')
        tensor_shape = conv_output_shape(
            tensor_shape, conv_par[_][1], conv_par[_][2], conv_par[_][3], conv_par[_][4])
        print(f'{tensor_shape}')
        print(f'--------Encoder.max{_+1}--------')
        tensor_shape = maxpool_output_shape(
            tensor_shape, max_par[_][0], max_par[_][1])
        print(f'{tensor_shape}')
    index = total_layers-NETWORK_DEPTH
    for _ in range(index, total_layers):
        print(f'--------Decoder.convt{_+1}--------')
        tensor_shape = conv_transpose_output_shape(
            tensor_shape, convt_par[_][1], convt_par[_][2], convt_par[_][3], convt_par[_][4], convt_par[_][5])
        print(f'{tensor_shape}')


if __name__ == '__main__':
    show_shape_transform(4, 8)
