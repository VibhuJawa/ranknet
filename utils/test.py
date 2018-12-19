import  math
from .ranking_metrics import ndcg_score

def test_step(model,ranknet_test_ds):
    ndgc_ls = []
    for i_batch, sample_batched in enumerate(ranknet_test_ds):
        label, data = sample_batched['y'], sample_batched['data']
        pred = model.predict(data)
        pred_ar = pred.squeeze(1).detach().numpy()
        label_ar = label.detach().numpy()
        ndgc_s = ndcg_score(label_ar, pred_ar)
        if not math.isnan(ndgc_s):
            ndgc_ls.append(ndgc_s)

        return sum(ndgc_ls) / len(ndgc_ls)


