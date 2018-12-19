from utils.dataloaders import RANKNET_TRAIN_DS,RANKNET_TEST_DS,transform_train,transform_test
from torch.utils.data import DataLoader
from utils.train import train_step
from utils.test import test_step
import time
import os
import argparse

import torch
# change this for different models
from models.ranknet_4_layers import RankNet
model_name = "ranknet_4_layers"



parser = argparse.ArgumentParser(description='Train Ranknet')
parser.add_argument('--train_dir', default="data/MQ2007/Fold1/train_json", type=str, help='train_dir')
parser.add_argument('--test_dir', default="data/MQ2007/Fold1/test_json", type=str, help='test_dir ')
parser.add_argument('--input_dim', default=46, type=int, help='input_dim')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--test_interval',default=5,type=int,help = 'interval b/w test runs')
parser.add_argument('--load_model',default=False)
parser.add_argument('--model_path',default=None)




args = parser.parse_args()


data_train_dir = args.train_dir
data_train_meta_csv = "{}/metafile.csv".format(data_train_dir)

data_test_dir = args.test_dir
data_test_meta_csv = "{}/metafile.csv".format(data_test_dir)

#input dimension
input_dim = args.input_dim

#batch size
batch_size = args.batch_size

#total_epochs
num_epochs = args.num_epochs

# test print interval
test_interval = args.test_interval


# model dir
model_dir =  "trained_models/{}_{}".format(model_name,data_train_dir)
os.makedirs(model_dir, exist_ok=True)



feats_to_drop = ['doc_id','inc','prob','qid','y']
feature_cols = [str(i) for i in range(1,input_dim+1)]



ranknet_train_ds = RANKNET_TRAIN_DS(data_train_meta_csv,data_train_dir,feature_cols,feats_to_drop,transform_train)


ranknet_dcg_test_dcg = RANKNET_TEST_DS(data_test_meta_csv,data_test_dir,feature_cols,feats_to_drop,transform_test)
ranknet_dcg_train_dcg = RANKNET_TEST_DS(data_train_meta_csv,data_train_dir,feature_cols,feats_to_drop,transform_test)

train_dataloader = DataLoader(ranknet_train_ds, batch_size=batch_size,shuffle=True, num_workers=4)



model = RankNet(input_dim)
if args.load_model:
    model.load_state_dict(torch.load(args.model_path))

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

epoch = 0

start_time = time.time()


for epoch in range(args.num_epochs+1):
    epoch_train_loss = train_step(model,train_dataloader,optimizer)
    elapsed_time = time.time() - start_time
    print("Epoch: {} Train Loss: {} Elapsed_Time: {}".format(epoch,epoch_train_loss,elapsed_time))
    if epoch%test_interval==0:
        epoch_test_dcg = test_step(model,ranknet_dcg_test_dcg)
        epoch_train_dcg = test_step(model,ranknet_dcg_train_dcg)
        elapsed_time = time.time() - start_time
        print("Epoch: {} Test DCG: {} Train DCG: {} Elapsed_Time: {}".format(epoch,epoch_test_dcg,epoch_train_dcg,elapsed_time))
        print("--"*30)
        model_file_path = "{}/test_dcg_{}_train_dcg_{}.model".format(model_dir,epoch_test_dcg,epoch_train_dcg)
        torch.save(model.state_dict(), model_file_path)



