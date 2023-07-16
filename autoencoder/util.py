import locale
import lzma
import os
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

username = os.getlogin()
home = str(Path.home())
loc = locale.getlocale()[0]

shell = subprocess.check_output(
    "env | grep SHELL=", shell=True).decode().strip()
if shell == "SHELL=/bin/bash":
    rc = ".bashrc"
else:
    rc = ".zshrc"
find = subprocess.check_output(
    f"grep -n 'crucio' {home}/{rc}", shell=True).decode().strip()
CRUCIO_DIR = find.split(":")[2]

if loc == 'zh_CN':
    media = '/media/'+username+'/DataDisks/Downloads'
    if os.path.isdir(media) and os.listdir(media):
        DOWNLOAD_DIR = media
    else:
        DOWNLOAD_DIR = '/media/'+username+'/PortableSSD/Downloads'
else:
    DOWNLOAD_DIR = '/data/'+username+'/crucio_downloads'

# png is lossless image compression (captured by camera)
IMAGE_EXT = '.png'
# Whether CUDA GPUs are used during evaluation
GPU_ENABLED = True
# Whether to use YUV color space during codec
YUV_ENABLED = True


def check_work_dir():
    '''
    Switch working directory to directory where py file (to be run) is located
    Calling check_work_dir function at beginning of py file running as main function
      corrects all relative paths in it
    Function is invalid if py file is called as a module by another main function
    In this case you should use absolute path 
      (configure crucio directory with command ./configure.sh 1)
    '''
    curr_dir = os.getcwd()
    file_path = sys.argv[0]
    # file_name = os.path.basename(file_path)
    # file_name = file_path.split('/')[-1]
    if file_path[0] == '/':
        work_dir = file_path
    else:
        if file_path[0] == '.' and file_path[1] == '/':
            work_dir = os.path.join(curr_dir, file_path[1:])
            # work_dir = curr_dir+file_path[1:]
        else:
            work_dir = os.path.join(curr_dir, file_path)
            # work_dir = curr_dir+'/'+file_path
    work_dir = os.path.dirname(work_dir)
    # work_dir = work_dir[:-(len(file_name))]
    if os.path.exists(work_dir):
        os.chdir(work_dir)


def convert_image_to_tensor(img_path, is_yuv, is_gpu):
    '''
    Convert an RGB image to a normalized tensor
    img_path -> Absolute path of image
    is_yuv -> Whether to use YUV color space
    is_gpu -> Whether tensor is moved to GPU
    '''
    if is_yuv:
        tensor_mode = 'YCbCr'
    else:
        tensor_mode = 'RGB'
    image = Image.open(img_path).convert(tensor_mode)
    tensor = transforms.ToTensor()(image)
    if is_gpu:
        tensor = tensor.cuda()
    return tensor


def convert_tensor_to_image(tensor, is_yuv):
    '''
    Converts normalization tensor to an RGB image
    tensor -> Output tensor from decoder
    is_yuv -> Whether to use YUV color space
    '''
    if is_yuv:
        array = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(np.transpose(array, (1, 2, 0)), mode='YCbCr')
        image = image.convert('RGB')
    else:
        image = transforms.ToPILImage(mode='RGB')(tensor.detach().cpu())
    return image


def save_compressed_data(input_path, compressed_data):
    '''
    input_path -> Absolute path of input image or video
    compressed_data -> Compressed data from encoder (tensor)
    Return absolute path to compressed data
    '''
    compressed_data = lzma.compress(pickle.dumps(compressed_data))
    data_name, ext = os.path.splitext(input_path)
    data_path = data_name + ".pkl"
    with open(data_path, 'wb') as f:
        f.write(compressed_data)
    return data_path


def load_compressed_data(data_path):
    '''
    data_path -> Absolute path of compressed data
    Return compressed data from encoder (tensor)
    '''
    with open(data_path, 'rb') as f:
        compressed_data = f.read()
    compressed_data = pickle.loads(lzma.decompress(compressed_data))
    if GPU_ENABLED:
        compressed_data = compressed_data.cuda()
    return compressed_data


def test_convert_function(image_path, is_yuv, is_gpu):
    '''
    Test correctness of images and tensor conversion functions
    '''
    check_work_dir()
    tensor = convert_image_to_tensor(image_path, is_yuv, is_gpu)
    converted_img = convert_tensor_to_image(tensor, is_yuv)
    converted_img.save('converted_image'+IMAGE_EXT)
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(image_path))
    plt.title('Test Image')
    plt.subplot(1, 2, 2)
    plt.imshow(plt.imread('converted_image'+IMAGE_EXT))
    plt.title('Converted Image')
    plt.show()
    os.remove('converted_image'+IMAGE_EXT)


def get_folder_size(folder_path):
    '''
    Calculates total size of all files in specified folder (KB)
    '''
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size/1024


def show_videos_difference(frame_number, video_path, reconstructed_path):
    '''
    Show image difference between original video and reconstructed video
    frame_number -> Number of frames in input video
    video_path -> Absolute path to original video directory
    reconstructed_path -> Absolute path to reconstructed video directory
    '''
    # Create a new image window
    fig = plt.figure(figsize=(12, 6))

    # Add subgraph for first row and set title
    for i in range(1, frame_number+1):
        str_num = "{:04d}".format(i)
        img_path = os.path.join(video_path, 'frame'+str_num + IMAGE_EXT)
        ax = fig.add_subplot(2, frame_number, i)
        ax.set_title('Frame'+str(i))
        ax.imshow(plt.imread(img_path))

    # Add subgraph for second row and set title
    for i in range(1, frame_number+1):
        str_num = "{:04d}".format(i)
        img_path = os.path.join(
            reconstructed_path, 'frame'+str_num + IMAGE_EXT)
        if os.path.exists(img_path):
            ax = fig.add_subplot(2, frame_number, frame_number + i)
            ax.set_title('Rec'+str(i))
            ax.imshow(plt.imread(img_path))

    # Adjust spacing between subgraphs
    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    # Show images
    plt.show()


def get_gpu_model():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], encoding='utf-8')
        gpu_model = result.strip().split('\n')[0]
        return gpu_model
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'Cannot get GPU model!'


if GPU_ENABLED is True:
    print('[Evaluation using CUDA GPU]')
    print(get_gpu_model())
    if torch.cuda.is_available() is False:
        print(
            'CUDA is not available due to mismatch between PyTorch version and CUDA version')
else:
    print('[Evaluation using CPU device]')
