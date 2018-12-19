import math

def train_step(model,train_dataloader,optimizer):
    epoch_loss_ls = []
    for i_batch, sample_batched in enumerate(train_dataloader):
        xi, ti = sample_batched['doc1']['data'], sample_batched['doc1']['y']
        xj, tj = sample_batched['doc2']['data'], sample_batched['doc2']['y']
        loss = model(xi, xj, ti, tj)
        total_loss = loss.mean()
        epoch_loss_ls.append(total_loss.item())
        model.zero_grad()
        total_loss.backward()
        optimizer.step()

    return sum(epoch_loss_ls) / len(epoch_loss_ls)
