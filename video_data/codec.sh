#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
error=1

filename="demo3.mp4"
basename="${filename%.*}"
framedir=$basename"_frame"
framename=$framedir/$basename"_"%04d.png

if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        if [ ! -d "$framedir" ]; then
            echo "Cannot find directory "$framedir
            exit
        fi
        ffmpeg -i $framename -crf 19 -c:v libx264 $filename
    elif [ "$1" == "0" ]; then
        error=0
        if [ ! -d "$framedir" ]; then
            mkdir "$framedir"
        fi
        ffmpeg -i $filename -vsync 0 $framename
    fi
fi
if [ "$error" == "1" ]; then
    echo "./codec.sh 1 Encoding video"
    echo "./codec.sh 0 Decoding video"
fi
