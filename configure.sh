#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
HOST=$(cat pipeline/config.json | jq -r '.host')
error=1

SERVER=0
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
    SERVER=1
fi

shell=$(env | grep SHELL=)
echo $shell
if [ $shell == "SHELL=/bin/bash" ]; then
    rc=.bashrc
else
    rc=.zshrc
fi
crucio=$(pwd)
# gedit ${HOME}/$rc
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        echo "Configure Crucio"
        if [ $(grep -c "export PYTHONPATH=\${PYTHONPATH}:$crucio" ${HOME}/$rc) -eq 0 ]; then
            sed -i "\$a export PYTHONPATH=\${PYTHONPATH}:$crucio" ${HOME}/$rc
        fi
        if [ "$SERVER" == "0" ]; then
            if [ $shell == "SHELL=/bin/bash" ]; then
                gnome-terminal -- /bin/sh -c 'source ${HOME}/$rc;exit'
            else
                gnome-terminal -- /bin/zsh -c 'source ${HOME}/$rc;exit'
            fi
        else
            echo "Please run source ${HOME}/$rc manually"
        fi
        echo "echo \${PYTHONPATH}"
    elif [ $1 == "0" ]; then
        error=0
        echo "Delete configure"
        crucio=$(echo "${crucio##*/}")
        sed -i "/$crucio/d" ${HOME}/$rc
        if [ "$SERVER" == "0" ]; then
            if [ $shell == "SHELL=/bin/bash" ]; then
                gnome-terminal -- /bin/sh -c 'source ${HOME}/$rc;exit'
            else
                gnome-terminal -- /bin/zsh -c 'source ${HOME}/$rc;exit'
            fi
        else
            echo "Please run source ${HOME}/$rc manually"
        fi
        echo "echo \${PYTHONPATH}"
    fi
fi
if [ $error == "1" ]; then
    echo "./configure.sh 1	Configure Crucio"
    echo "./configure.sh 0	Delete configure"
fi
