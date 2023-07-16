# Crucio: End-to-End Spatio-Temporal Redundancy Elimination for Online Video Analytics Acceleration

This repository hosts the prototype implementation of our paper *Crucio: End-to-End Spatio-Temporal Redundancy Elimination for Online Video Analytics Acceleration*.

## Prerequisites

Python 3.10  
CUDA 11.8.0  
PyTorch 2.0.1  
OpenCV 4.8.0  
ffmpeg 7:4.4.2-0ubuntu0.22.04.1  
torchmetrics 1.0.0  
pytorch-msssim 1.0.0  
pycocotools 2.0.6  
screen 4.9.0-1  
openssh-client 1:8.9p1-3ubuntu0.1  
openssh-server 1:8.9p1-3ubuntu0.1  
sshpass 1.09-1  
jq 1.6-2.1ubuntu3  

## Install Instructions

To deploy our code, first execute  
``cd crucio;./configure.sh 1``  
to configure environment variables.

Before running *Crucio*, you need to download dataset *youtube-objects* and *val2017* to the directory *DOWNLOAD_DIR* identified by ``autoencoder/util.py``,
Similarly, download DNN model weights corresponding to analytics task to this directory, and execute ``./dnn_model/weights.sh`` to install them.

Now, we can execute the following commands to train *GRU-CNN Encoder* and *CNN Decoder* separately  
``python3 autoencoder/train3d.py``  
``python3 batch_handler/gru_train.py``  
The trained codec weights are automatically saved in the directory *DOWNLOAD_DIR*.

Note that *Batch Scheduler* uses the default fitting results on our device, to perform a new fit visit *offline_profiling* function in ``batch_handler/scheduler.py``.

After the training is complete, you can run the following command to launch *Crucio*  
``python3 preheat.py``

To enable real-time network transmission, execute ``pipeline/client.py`` and ``pipeline/server.py``.
