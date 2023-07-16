#!/bin/bash
input=${HOME}/.local/lib/python3.*/site-packages/torchvision/models/detection
input=$(echo ${input})
if [ ! -d "${input}" ]; then
    echo "${input}/ dose not exist"
    exit
fi
echo ${input}
error=1
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        echo "[1]"
        sed -i "s/def forward(self, images, targets=None):/def forward(self, images, targets=None, RPN=False):/g" ${input}/generalized_rcnn.py
        echo "[2]"
        sed -i "s/\
        proposals, proposal_losses = self.rpn(images, features, targets)/\
        proposals, proposal_losses = self.rpn(images, features, targets, RPN)\\n\
        if RPN:\\n\
            proposals, scores = proposals/g" ${input}/generalized_rcnn.py
        echo "[3]"
        sed -i "s/^\
            return self.eager_outputs(losses, detections)/\
            if RPN:\\n\
                from torchvision.models.detection.transform import resize_boxes\\n\
                for i, (pred, im_s, o_im_s) in enumerate(zip(proposals, images.image_sizes, original_image_sizes)):\\n\
                    boxes = resize_boxes(pred, im_s, o_im_s)\\n\
                    proposals[i] = boxes\\n\
                return self.eager_outputs(losses, detections), proposals, scores\\n\
            else:\\n\
                return self.eager_outputs(losses, detections)/g" ${input}/generalized_rcnn.py
        echo "Modify generalized_rcnn.py"
        echo "[1]"
        find=$(grep -n "RPN=False," ${input}/rpn.py | cut -d ":" -f 1)
        if [ -z "${find}" ]; then
            find=$(grep -n "targets: Optional\[List\[Dict\[str, Tensor\]\]\] = None," ${input}/rpn.py | cut -d ":" -f 1)
            if [ "${find}" ]; then
                find=$((find + 1))
                sed -i "${find}i \    \    RPN=False," ${input}/rpn.py
            fi
        fi
        echo "[2]"
        find=$(grep -n "boxes = boxes, scores" ${input}/rpn.py | cut -d ":" -f 1)
        if [ -z "${find}" ]; then
            sed -i "s/\
        return boxes, losses/\
        if RPN:\\n\
            boxes = boxes, scores\\n\
        return boxes, losses/g" ${input}/rpn.py
        fi
        echo "Modify rpn.py"
    elif [ $1 == "0" ]; then
        error=0
        echo "[1]"
        sed -i "s/def forward(self, images, targets=None, RPN=False):/\
def forward(self, images, targets=None):/g" ${input}/generalized_rcnn.py
        echo "[2]"
        find=$(grep -n "proposals, proposal_losses = self.rpn(images, features, targets, RPN)" ${input}/generalized_rcnn.py | cut -d ":" -f 1)
        if [ "${find}" ]; then
            find=$((find + 1))
            for i in {1..2}; do
                sed -i "${find}d" ${iusernamenput}/generalized_rcnn.py
            done
            sed -i "s/proposals, proposal_losses = self.rpn(images, features, targets, RPN)/\
proposals, proposal_losses = self.rpn(images, features, targets)/g" ${input}/generalized_rcnn.py
        fi
        echo "[3]"
        find=$(grep -n "from torchvision.models.detection.transform import resize_boxes" ${input}/generalized_rcnn.py | cut -d ":" -f 1)
        if [ "${find}" ]; then
            for i in {1..7}; do
                sed -i "${find}d" ${input}/generalized_rcnn.py
            done
            sed -i "s/if RPN:/\
return self.eager_outputs(losses, detections)/g" ${input}/generalized_rcnn.py
        fi
        echo "Restore generalized_rcnn.py"
        echo "[1]"
        sed -i "/RPN=False,/d" ${input}/rpn.py
        echo "[2]"
        find=$(grep -n "boxes = boxes, scores" ${input}/rpn.py | cut -d ":" -f 1)
        if [ "${find}" ]; then
            find=$(grep -n "return boxes, losses" ${input}/rpn.py | cut -d ":" -f 1)
            if [ "${find}" ]; then
                for i in {1..2}; do
                    find=$((find - 1))
                    sed -i "${find}d" ${input}/rpn.py
                done
            fi
        fi
        echo "Restore rpn.py"
    fi
fi
if [ $error == "1" ]; then
    echo "./prerequisite.sh 1	Modify PyTorch (generalized_rcnn.py and rpn.py)"
    echo "./prerequisite.sh 0	Restore PyTorch (generalized_rcnn.py and rpn.py)"
fi
