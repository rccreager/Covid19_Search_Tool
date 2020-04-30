#!/bin/sh
bert-serving-start -num_worker=$1 -model_dir /uncased_L-12_H-768_A-12 -max_seq_len None
