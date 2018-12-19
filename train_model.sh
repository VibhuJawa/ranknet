#!/bin/sh
# datadir
data_dir=data/MQ2007/Fold2

python base_train.py --input_dim 46 --train_dir $data_dir/train_json --test_dir $data_dir/test_json --num_epochs 10 --test_interval 5
