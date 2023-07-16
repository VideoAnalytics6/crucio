#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
HOST=$(cat config.json | jq -r '.host')
USERNAME=$(cat config.json | jq -r '.username')
DOWNLOAD_DIRS="/data/$USERNAME/crucio_downloads"
PASSWORD=$(cat config.json | jq -r '.password')

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
if [ "$server_IP" == "$HOST" ]; then
    echo -e "\e[31mNote: Run client.sh on client!\e[0m"
    exit
fi

input_path=$(../video_data/download.sh)
echo ${input_path}

if [ ! -d "${input_path}" ]; then
    echo "Directory ${input_path} does not exist"
    exit
fi

sudo apt install sshpass -y
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d /home/$USERNAME/.cache/torch ]"; then
    scp -r ${HOME}/.cache/torch $USERNAME@$HOST:/home/$USERNAME/.cache/torch
fi
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d /data/$USERNAME ]"; then
    echo "/data/$USERNAME does not exist in host ${HOST}"
    exit
fi
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d $DOWNLOAD_DIRS ]"; then
    scp -r $input_path $USERNAME@$HOST:$DOWNLOAD_DIRS
fi
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ -d $DOWNLOAD_DIRS/checkpoints ]"; then
    ssh $USERNAME@$HOST "rm -rf $DOWNLOAD_DIRS/checkpoints"
fi
