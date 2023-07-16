#!/bin/bash
if [ "$LANG" == "zh_CN.UTF-8" ]; then
    media="/media/${USER}/DataDisks/Downloads"
    if [[ -d "${media}" && -n "$(ls -A ${media})" ]]; then
        DOWNLOAD_DIR="${media}"
    else
        DOWNLOAD_DIR="/media/${USER}/PortableSSD/Downloads"
    fi
else
    DOWNLOAD_DIR="/data/${USER}/crucio_downloads"
fi
echo ${DOWNLOAD_DIR}
