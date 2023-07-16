#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
HOST=$(cat config.json | jq -r '.host')

inet=$(ifconfig | grep -A 1 "eno1")
OLD_IFS="$IFS"
IFS=" "
inets=($inet)
IFS="$OLD_IFS"
let "i=0"
for var in ${inets[@]}; do
    if [ $var == "inet" ]; then
        break
    fi
    let "i=i+1"
done
let "i=i+1"
server_IP=${inets[$i]}
server_IP=$(echo "$server_IP" | sed 's/ //g')
if [ "$server_IP" != "$HOST" ]; then
    echo -e "\e[31mNote:Run server.sh on server!\e[0m"
    exit
fi

if [ ! -d "/data/${USER}" ]; then
    sudo mkdir /data/${USER}
fi
sudo chown -R ${USER} /data/${USER}
sudo apt install python3.8 python3-pip python3-socks -y
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --config python3
sudo apt install libopencv-dev python3-opencv -y
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
python3 -m pip install pysocks opencv-python testresources
python3 -c "import cv2; print(cv2.__version__)"
# python3 -m pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
# CUDA driver version is 11.5
nvidia-smi
python3 -m pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0 -f https://DOWNLOAD_DIR.pytorch.org/whl/torch_stable.html
python3 -m pip install matplotlib
echo "Pytorch CUDA"
echo "import torch;print(torch.cuda.is_available())" | python3
