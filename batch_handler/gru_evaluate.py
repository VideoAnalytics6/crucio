import os

import matplotlib.pyplot as plt
import torch

from autoencoder.dataset import (CNN_FRAME_NUM, IMAGE_EXT, VIDEO_DIR,
                                 load_video_to_tensor)
from batch_handler.gru_filter import get_filter, scores_to_selects


def show_filter_results(number, video_path, selects):
    '''
    Display video filtering (1 for keyframe and 0 for filtered frame)
    number -> Number of frames in video
    video_path -> Absolute path to video directory
    '''
    # Create a new image window
    fig = plt.figure(figsize=(12, 6))

    # Add subgraph for first row and set title
    for i in range(1, number+1):
        str_num = "{:04d}".format(i)
        img_path = os.path.join(video_path, 'frame'+str_num + IMAGE_EXT)
        ax = fig.add_subplot(1, number, i)
        ax.set_title('Frame'+str(i)+'='+str(int(selects[i-1])))
        ax.imshow(plt.imread(img_path))

    # Adjust spacing between subgraphs
    plt.subplots_adjust(wspace=0.2)

    # Show images
    plt.show()


# load trained GRU encoder
extractor, gru = get_filter('eval', True)

# GRU filter test video
video_path = VIDEO_DIR+'/aeroplane_0001_039'
video_tensor = load_video_to_tensor(video_path).unsqueeze(0)
with torch.no_grad():
    features, size = extractor(video_tensor)
    scores = gru(features)

# Analyse video filter results
selects = scores_to_selects(scores).detach().cpu().numpy()
print(selects)
show_filter_results(CNN_FRAME_NUM, video_path, selects[0])
