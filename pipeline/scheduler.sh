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
    echo -e "\e[31mNote: Run scheduler.sh on server!\e[0m"
    exit
fi
