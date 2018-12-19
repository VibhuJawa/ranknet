import torch.nn as nn
import torch
import torch.nn.functional as F

# Model.
class RankNet(nn.Module):
    def __init__(self, input_dim):
        super(RankNet, self).__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 1)

    def single_forward(self, x):
        return self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x)))))))

    def forward(self, x_i, x_j, t_i, t_j):
        s_i = self.single_forward(x_i)
        s_j = self.single_forward(x_j)
        s_diff = s_i - s_j
        s_diff = s_diff.squeeze(1)
        S_ij = torch.zeros(size=t_i.shape)
        pos_mask = t_i > t_j
        neg_mask = t_i < t_j
        equal_mask = t_i == t_j
        S_ij[pos_mask] = 1
        S_ij[neg_mask] = -1
        S_ij[equal_mask] = 0

        loss = (1 - S_ij) * s_diff / 2.0 + torch.log(1 + torch.exp(-s_diff))
        return loss

    def predict(self, x):
        return self.single_forward(x)

