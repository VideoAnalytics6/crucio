import glob
import os
import re
import subprocess
import time
from multiprocessing.pool import ThreadPool

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from autoencoder.util import (CRUCIO_DIR, DOWNLOAD_DIR, GPU_ENABLED, IMAGE_EXT,
                              YUV_ENABLED, convert_image_to_tensor,
                              convert_tensor_to_image, test_convert_function)

# https://cocodataset.org/#download
IMAGE_DIR = DOWNLOAD_DIR+'/val2017'
# https://data.vision.ee.ethz.ch/cvl/youtube-objects/
VIDEO_DIR = DOWNLOAD_DIR+'/youtube-objects'
# Resolution is limited by GPU memory
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
# Minimum frame number supported by 3D-CNN structure
MIN_FRAME_NUM = 5
# Frame number used when training 3D-CNN Encoder/Decoder
CNN_FRAME_NUM = 8
assert CNN_FRAME_NUM >= MIN_FRAME_NUM
# Maximum frame number in dataset
find = subprocess.check_output(
    f"grep -n 'MAX_FRAME_NUM=' {CRUCIO_DIR}/video_data/prepare.sh", shell=True).decode().strip()
MAX_FRAME_NUM = int(find.split("=")[1])


class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = glob.glob(os.path.join(root_dir, '*'+IMAGE_EXT))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = convert_image_to_tensor(img_name, YUV_ENABLED, GPU_ENABLED)
        return image


def load_video_to_tensor(video_path, start_num=1, length=CNN_FRAME_NUM, return_list=False, check=False):
    '''
    Loads multiple consecutive video frames and returns a tensor
    video_path -> Absolute path to video directory
    start_num -> Number of first frame (starting from 1)
    length -> Number of frames to load
    return_list -> Whether to return a list of tensors
    check -> True only when checking dataset
    '''
    assert start_num >= 1
    if check:
        files = glob.glob(os.path.join(video_path, '*'+IMAGE_EXT))
        file_name = os.path.basename(files[0])
        string = os.path.splitext(file_name)[0]
        assert re.match(r'frame\d{4}', string)
        start_num = 1
        length = len(files)
    imgs = []
    curr_length = 0
    curr_num = start_num
    while curr_length < length:
        if curr_num > MAX_FRAME_NUM:
            break
        str_num = "{:04d}".format(curr_num)
        img_path = os.path.join(video_path, 'frame'+str_num+IMAGE_EXT)
        if os.path.exists(img_path):
            img = convert_image_to_tensor(img_path, YUV_ENABLED, GPU_ENABLED)
            imgs.append(img)
            curr_length += 1
        curr_num += 1
    if return_list:
        tensor = imgs, curr_num
    else:
        tensor = torch.stack(imgs, dim=1)
    return tensor


def save_tensor_to_video(video_path, tensor):
    '''
    Saves a tensor to specified video directory
    video_path -> Absolute path to video directory
    tensor -> A tensor from decoder
    '''
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    len = tensor.shape[1]
    for _ in range(len):
        img = convert_tensor_to_image(tensor[:, _], YUV_ENABLED)
        str_num = "{:04d}".format(_+1)
        img.save(os.path.join(video_path, 'frame'+str_num+IMAGE_EXT))


class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_list = glob.glob(os.path.join(root_dir, "*[0-9]*"))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        video = load_video_to_tensor(video_path)
        return video


def test_video_dataset():
    video_list = glob.glob(os.path.join(VIDEO_DIR, "*[0-9]*"))
    index = 1
    for video_name in video_list:
        tensor = load_video_to_tensor(video_name, check=True)
        print(f'num {index} path {video_name} tensor {tensor.shape}')
        index += 1


def convert_image(filename):
    format = IMAGE_EXT[1:].upper()
    try:
        img = Image.open(filename)
        # If image is already required size and format, processing is skipped
        if img.format == format and img.size == (IMAGE_WIDTH, IMAGE_HEIGHT):
            img.close()
            return

        # Resize image and save to desired format
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        new_filename = os.path.splitext(filename)[0] + IMAGE_EXT
        img.save(new_filename, format)
        img.close()

        # Delete original file
        if not filename.endswith(IMAGE_EXT):
            os.remove(filename)

    except (OSError, UnidentifiedImageError):
        # Delete other files or invalid images
        os.remove(filename)


def preprocess_image_dataset(input_dir):
    start = time.time()
    if os.path.exists(input_dir):
        # Gets filenames of all images in input directory
        image_filenames = [os.path.join(input_dir, file)
                           for file in os.listdir(input_dir)]
        frame_number = len(image_filenames)
        # Use thread pools to process all images in parallel
        pool = ThreadPool()
        pool.map(convert_image, image_filenames)
        pool.close()
        pool.join()
        print(f'Preprocessing of {frame_number} images completed')
    else:
        print(f'Directory {input_dir} does not exist')
    end = time.time()
    print(f'Running time {(end-start)/60}m')


def preprocess_video_dataset(input_dir):
    start = time.time()
    subprocess.call(['bash', CRUCIO_DIR+'/video_data/prepare.sh'])
    if os.path.exists(input_dir):
        video_filenames = []
        image_filenames = []
        for f in os.listdir(input_dir):
            video_filename = os.path.join(input_dir, f)
            if os.path.isdir(video_filename):
                video_filenames.append(video_filename)
                image_filenames += [os.path.join(video_filename, file)
                                    for file in os.listdir(video_filename)]
        video_number = len(video_filenames)
        pool = ThreadPool()
        pool.map(convert_image, image_filenames)
        pool.close()
        pool.join()
        print(f'Preprocessing of {video_number} videos completed')
    else:
        print(f'Directory {input_dir} does not exist')
    end = time.time()
    print(f'Running time {(end-start)/60}m')


if __name__ == '__main__':
    # test_convert_function(IMAGE_DIR+'/000000013291' +
    #                       IMAGE_EXT, YUV_ENABLED, GPU_ENABLED)
    preprocess_image_dataset(IMAGE_DIR)
    preprocess_video_dataset(VIDEO_DIR)
    test_video_dataset()
