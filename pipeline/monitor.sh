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
if [ "$server_IP" == "$HOST" ]; then
    echo -e "\e[31mNote: Run monitor.sh on client!\e[0m"
    exit
fi

# Number of time slots should ensure that client can handle all samples (datalength)
slot_number=6
# Time slot and frame rate should match video_data/prepare.sh/MAX_FRAME_NUM (45)
# s
time_slot=5
# fps
min_f=15
max_f=60
diff_f=$((max_f - min_f + 1))
# Kbps
net_device=$(iwconfig 2>/dev/null | awk '/^[a-zA-Z0-9]+/{print $1}')
echo "Get a wireless card ${net_device}"
min_bw=512
max_bw=2048
diff_bm=$((max_bw - min_bw + 1))
# Computing benchmark bandwidth
bench_bw=1024
echo "Benchmark network bandwidth is ${bench_bw}Kbps"

max_packet=1500
bench_delay=$(echo "${max_packet}*8/1024/${bench_bw}" | bc -l | awk '{printf "%.2f\n", $1}')
sudo tc qdisc add dev ${net_device} root tbf rate ${bench_bw}kbit burst ${bench_bw}kbit latency ${bench_delay}s
for i in $(seq 1 ${slot_number}); do
    echo "${time_slot}s time slot [${i}/${slot_number}]"
    f=$((RANDOM % ${diff_f} + ${min_f}))
    echo "Current frame rate is ${f}fps"
    bandwidth=$((RANDOM % ${diff_bm} + ${min_bw}))
    delay=$(echo "${max_packet}*8/1024/${bandwidth}" | bc -l | awk '{printf "%.2f\n", $1}')
    echo "Current bandwidth is ${bandwidth}Kbps"
    sudo tc qdisc change dev ${net_device} root tbf rate ${bandwidth}kbit burst ${bandwidth}kbit latency ${delay}s
    sleep ${time_slot}
done
echo "All restrictions on network bandwidth have been lifted"
sudo tc qdisc del dev ${net_device} root
