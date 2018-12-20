import math
import torch

is_cuda_available = torch.cuda.is_available()

def train_step(model,train_dataloader,optimizer):
    epoch_loss_ls = []
    for i_batch, sample_batched in enumerate(train_dataloader):
        if is_cuda_available:
           xi, ti = sample_batched['doc1']['data'].cuda(), sample_batched['doc1']['y'].cuda()
           xj, tj = sample_batched['doc2']['data'].cuda(), sample_batched['doc2']['y'].cuda()
        else:
           xi, ti = sample_batched['doc1']['data'], sample_batched['doc1']['y']
           xj, tj = sample_batched['doc2']['data'], sample_batched['doc2']['y']

        loss = model(xi, xj, ti, tj)
        total_loss = loss.mean()
        model.zero_grad()
        epoch_loss_ls.append(total_loss.item())
        total_loss.backward()
        optimizer.step()

    return sum(epoch_loss_ls) / len(epoch_loss_ls)
