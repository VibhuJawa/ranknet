#!/bin/sh

# datadir
# data_dir=data/MQ2007/Fold2
data_dir=data/MSLR-WEB10K/Fold1

#train
python utils/transform_dataset.py --data_file $data_dir/train.txt --output_dir $data_dir/train_json

# test
python utils/transform_dataset.py --data_file $data_dir/test.txt --output_dir $data_dir/test_json
