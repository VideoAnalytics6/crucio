#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
input_path=$(../video_data/download.sh)
echo ${input_path}

target_path="${HOME}/.cache/torch/hub"
if [ ! -d "${input_path}/checkpoints" ]; then
    echo "${input_path}/checkpoints dose not exist"
    exit
fi
if [ ! -d "${target_path}" ]; then
    echo "${target_path} dose not exist"
    exit
fi
if [ -d "${target_path}/checkpoints" ]; then
    rm -rf "${target_path}/checkpoints"
fi
cp -r "${input_path}/checkpoints" ${target_path}
