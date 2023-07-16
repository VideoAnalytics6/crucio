import os

import torch
from dataset import CNN_FRAME_NUM

from autoencoder.dataset import (VIDEO_DIR, load_video_to_tensor,
                                 save_tensor_to_video)
from autoencoder.network3d import get_networks3d
from autoencoder.util import (get_folder_size, load_compressed_data,
                              save_compressed_data, show_videos_difference)

# Load trained encoder and decoder
encoder3d, decoder3d = get_networks3d('eval', True)

# Encoder compresses test video
video_path = VIDEO_DIR+'/aeroplane_0001_033'
video_size = get_folder_size(video_path)
print(f"Size of original video {video_path} is {video_size:.4f} KB")
video_tensor = load_video_to_tensor(video_path).unsqueeze(0)
with torch.no_grad():
    compressed_data = encoder3d(video_tensor)

# Save compressed data
data_path = save_compressed_data(video_path, compressed_data)
data_size = os.path.getsize(data_path)/1024
print(f"Size of compressed data {data_path} is {data_size:.4f} KB")

# Load compressed data
compressed_data = load_compressed_data(data_path)

# Extract compressed data to images
with torch.no_grad():
    decoded_tensor = decoder3d(compressed_data)

# Save images to video
reconstructed_path = video_path+'_rec'
save_tensor_to_video(reconstructed_path, decoded_tensor[0])
reconstructed_size = get_folder_size(reconstructed_path)
print(
    f"Size of reconstructed video {reconstructed_path} is {reconstructed_size:.4f} KB")

# Show images in video
show_videos_difference(CNN_FRAME_NUM, video_path, reconstructed_path)
