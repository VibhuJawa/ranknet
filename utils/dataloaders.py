#   [1] https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf
#   [2] https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader


is_cuda = torch.cuda.is_available()

print("Cuda availabilty = {}".format(is_cuda))

def transform_train(q_sample, feature_cols,cols_to_drop):
    """
        input dataframe
        transforms datafram into tensor
    """

    if is_cuda:
      label_tensor = torch.tensor(int(q_sample['y'])).cuda()
      data_tensor = torch.tensor(q_sample[feature_cols].values.astype('float')).float().cuda()
    else:
      label_tensor = torch.tensor(int(q_sample['y']))
      data_tensor = torch.tensor(q_sample[feature_cols].values.astype('float')).float()
    return {'y': label_tensor, 'data': data_tensor}


class RANKNET_TRAIN_DS(Dataset):
    """Document Ranking Dataset."""

    def __init__(self, csv_file, root_dir,feature_cols,feats_to_drop, transform=None):
        """
        Args:
            text_file (string): Path to the txt file with q_id.
            root_dir (string): Directory with all the query_details.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.feats_to_drop = feats_to_drop
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.meta_file)

    def __getitem__(self, idx):
        q_fname = os.path.join(self.root_dir, str(self.meta_file.iloc[idx]['qid']))
        q_data = pd.read_csv("{}.csv".format(q_fname))
        i1, i2 = np.random.choice(len(q_data), 2)
        z1, z2 = q_data.iloc[i1], q_data.iloc[i2]
        sample = {'doc1': transform_train(z1, self.feature_cols,self.feats_to_drop),
                  'doc2': transform_train(z2, self.feature_cols,self.feats_to_drop)}
        return sample



def transform_test(q_sample_ls, feature_cols,cols_to_drop):
    """
        input dataframe
        transforms datafram into tensor
    """
    if is_cuda:
       label_tensor_ls = torch.tensor(np.asarray([q_sample['y'] for q_sample in q_sample_ls])).cuda()
       data_tensor_ls = torch.tensor( \
        np.asarray([q_sample[feature_cols].values.astype('float') for q_sample in q_sample_ls])).float().cuda()
    else:
      label_tensor_ls = torch.tensor(np.asarray([q_sample['y'] for q_sample in q_sample_ls]))

      data_tensor_ls = torch.tensor( \
        np.asarray([q_sample[feature_cols].values.astype('float') for q_sample in q_sample_ls])).float()

    return {'y': label_tensor_ls, 'data': data_tensor_ls}


class RANKNET_TEST_DS(Dataset):
    """Document Ranking Dataset."""

    def __init__(self, csv_file, root_dir, feature_cols,feats_to_drop,transform=None):
        """
        Args:
            csv_file (string): Path to the txt file with q_id.
            root_dir (string): Directory with all the query_details.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.feats_to_drop = feats_to_drop
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.meta_file)

    def __getitem__(self, idx):
        q_fname = os.path.join(self.root_dir, str(self.meta_file.iloc[idx]['qid']))
        q_data = pd.read_csv("{}.csv".format(q_fname))
        z_ls = [q_data.iloc[i] for i in range(len(q_data))]
        sample_ls = transform_test(z_ls, self.feature_cols,self.feats_to_drop)
        return sample_ls




