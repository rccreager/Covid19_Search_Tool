#!/bin/sh

#stupid hacky fix to get GPU working with containers on EC2
#export NVIDIA_VISIBLE_DEVICES=all
#export NVIDIA_DRIVER_CAPABILITIES=compute,utility
#python3 /bert/start_gpu_sess.py

bert-serving-start -num_worker=$1 -model_dir=/uncased_L-12_H-768_A-12 -max_seq_len=$2 -device_map=0
