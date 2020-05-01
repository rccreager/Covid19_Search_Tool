#!/bin/sh
python3 /bert/test.py
bert-serving-start -num_worker=$1 -model_dir=/uncased_L-12_H-768_A-12 -max_seq_len=$2 -device_map=0 -num_worker=1
