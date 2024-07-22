"""
This script handles the training process process for various Transformer-based models.
It includes features for training, evaluation, & inference.
"""
__author__ = "AUTHOR_NAME_PLACEHOLDER"

from utils import *
from itertools import compress
import gc
import numpy as np
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans as th_Kmeans
import torch
import math


def train_teacherEnforce(model, optimizer, train_dataloader, PAD_IDX,  BOS_IDX, EOS_IDX, DEVICE="cpu"):
    """
    Trains a seq2seq model with teacher forcing using MSE loss and Pearson correlation as a metric,
    iterating through batches, calculating loss & gradients, and updating model parameters.
    """

    model.train()
    losses = 0
    r = 0
 
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :, :].to(DEVICE)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src=src,
            tgt=tgt,
            PAD_IDX=PAD_IDX,
            device=DEVICE,
        )

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(DEVICE), tgt_mask.to(DEVICE), \
                src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)
        
        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask
            ).to(DEVICE)
        
        optimizer.zero_grad()
        tgt_out = tgt[1:, :, :].to(DEVICE)

        loss = mse_loss(
            logits.reshape(-1),
            tgt_out.reshape(-1), 
            ignored_indices=torch.Tensor([PAD_IDX, BOS_IDX, EOS_IDX]).to(DEVICE), reduction='mean'
            )
        
        r += pearsonr_corr(
            logits.reshape(-1),
            tgt_out.reshape(-1),
            ignored_indices=torch.Tensor([PAD_IDX, BOS_IDX, EOS_IDX]).to(DEVICE)
            )
        
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader)), r / len(list(train_dataloader))


def evaluate(model, val_dataloader, PAD_IDX,  BOS_IDX, EOS_IDX, DEVICE):
    """ 
        This function test the performance based on teacher training.

        Paramters:
        - model: The model to be evaluated
        - val_dataloader: Dataloader for the validation data
        - PAD_IDX: Padding index used in the data
        - BOS_IDX: Beginning of Sequence token used in the data
        - EOS_IDX: End of sequence token used in the data
    """

    model.eval()
    losses = 0
    r = 0

    with torch.no_grad():
        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_input = tgt[:-1, :, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src,
                tgt_input,
                PAD_IDX,
                DEVICE
                )
            
            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask
                )

            tgt_out = tgt[1:, :, :]

            loss = mse_loss(
                logits.reshape(-1),
                tgt_out.reshape(-1),
                ignored_indices=torch.Tensor([PAD_IDX, BOS_IDX, EOS_IDX]).to(DEVICE), reduction='mean'
                )
            
            r += pearsonr_corr(
                logits.reshape(-1),
                tgt_out.reshape(-1),
                ignored_indices=torch.Tensor([PAD_IDX, BOS_IDX, EOS_IDX]).to(DEVICE)
                )

            losses += loss.item()

    return losses / len(list(val_dataloader)), r / len(list(val_dataloader))


def run_batch(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt):
    """
    Performs single sample inference with a Transformer model,
    accumulating encoder outputs (memories) for later analysis.

    """

    max_len = max_lens[sample_n]
    ys = torch.ones(1, 1, n_tasks).fill_(BOS_IDX).to(DEVICE)
    memories = []

    for i in range(max_len-1):
        src_ = src[0:(i+1), sample_n:(sample_n+1), :]
        src_mask = torch.zeros((src_.shape[0], src_.shape[0]),device=DEVICE).type(torch.bool) 
        tgt_mask = torch.zeros((ys.shape[0], ys.shape[0]),device=DEVICE).type(torch.bool) 

        memory = transformer.encode(src_, src_mask).to(DEVICE)
        memories += [torch.clone(memory).detach().cpu()]
        out = transformer.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        pred = transformer.generator(out[:, -1, :])
        pred = pred.reshape([pred.shape[0], 1, pred.shape[1]])
        ys = torch.cat([ys, pred], dim=0)
    

    tgt_noBOSEOS = tgt[1:max_len, sample_n:(sample_n+1), :].detach().cpu().numpy()
    ys_noBOSEOS = ys[1:, :, :].detach().cpu().numpy()

    # Empty the memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return (
        ys[1:, :, :],
        tgt[1:max_len,sample_n:(sample_n+1), :],
        ys_noBOSEOS,
        tgt_noBOSEOS,
        torch.cat(memories).detach().cpu().numpy()
    )


def run_batch_sliding(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, sl):
    """
        Perfroms a single sample inference with a Transformer model using sliding window on soource 
        and target sequences for memory management
    """

    max_len = max_lens[sample_n]
    ys = torch.ones(1, 1, n_tasks).fill_(BOS_IDX).to(DEVICE) #.type(torch.long)
    memories = []

    for i in range(max_len-1):
        ys_sl = ys if ys.shape[0]<=sl else ys[-sl:,:,:]
        src_ = src[0:(i+1), sample_n:(sample_n+1), :]
        src_sl = src_ if src_.shape[0]<=sl else src_[-sl:, :,:]

        src_mask = torch.zeros((src_sl.shape[0], src_sl.shape[0]),device=DEVICE).type(torch.bool)
        tgt_mask = torch.zeros((ys_sl.shape[0], ys_sl.shape[0]),device=DEVICE).type(torch.bool) 
        memory = transformer.encode(src_sl, src_mask).to(DEVICE)
        memories += [torch.clone(memory).detach().cpu()]
        out = transformer.decode(ys_sl, memory, tgt_mask)
        out = out.transpose(0, 1)
        pred = transformer.generator(out[:, -1, :])
        pred = pred.reshape([pred.shape[0], 1, pred.shape[1]])

        ys = torch.cat([ys, pred], dim=0)
        
    tgt_noBOSEOS = tgt[1:max_len, sample_n:(sample_n+1), :].detach().cpu().numpy()
    ys_noBOSEOS = ys[1:, :, :].detach().cpu().numpy()
    torch.cuda.empty_cache()
    gc.collect()
    
    return ys[1:, :, :], tgt[1:max_len, sample_n:(sample_n+1), :], ys_noBOSEOS, tgt_noBOSEOS, torch.cat(memories).detach().cpu().numpy()



def run_batch_km(km, sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt):
    """
    Performs ssingle sample inference using a Transformer model and KMeans clustering on encoder to guide the decoder.

    """

    max_len = max_lens[sample_n]
    ys = torch.ones(1, 1, n_tasks).fill_(BOS_IDX).to(DEVICE) #.type(torch.long)

    for i in range(max_len-1):
        src_ = src[0:(i+1), sample_n:(sample_n+1), :].to(DEVICE)
        src_mask = torch.zeros((src_.shape[0], src_.shape[0]),device=DEVICE).type(torch.bool)
        tgt_mask = torch.zeros((ys.shape[0], ys.shape[0]),device=DEVICE).type(torch.bool)
        memory = transformer.encode(src_, src_mask).to(DEVICE)
        
        x = memory.squeeze(1).detach().cpu().numpy()
        centers = torch.Tensor(km.cluster_centers_[km.predict(x), :]).unsqueeze(1).to(DEVICE)
        out = transformer.decode(ys, centers, tgt_mask)
        out = out.transpose(0, 1)
        pred = transformer.generator(out[:, -1, :])
        pred = pred.reshape([pred.shape[0], 1, pred.shape[1]])
        ys = torch.cat([ys, pred], dim=0)

    tgt_noBOSEOS = tgt[1:max_len, sample_n:(sample_n+1), :]
    ys_noBOSEOS = ys[1:, :, :]

    return ys[1:, :, :], tgt[1:max_len, sample_n:(sample_n+1), :], ys_noBOSEOS, tgt_noBOSEOS


def run_batch_cl(centers, sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC, src, tgt):
    """
    Performs single ssample inference using a Transformer and clustering,
    routing encoder outputs to specific decoder layers based on 

    """

    max_len = max_lens[sample_n]
    ys = torch.ones(1, 1, n_tasks).fill_(BOS_IDX).to(DEVICE)
    q = []

    for i in range(max_len-1):
        src_ = src[0:(i+1), sample_n:(sample_n+1), :].to(DEVICE)
        src_mask = torch.zeros((src_.shape[0], src_.shape[0]),device=DEVICE).type(torch.bool) 
        tgt_mask = torch.zeros((ys.shape[0], ys.shape[0]),device=DEVICE).type(torch.bool)
        memory = trans_IDEC.transformer.encode(src_, src_mask).to(DEVICE)

        z = memory.squeeze(1)
        tmp_q = trans_IDEC(z)
        q += [tmp_q.cpu()]
        clus_pred = tmp_q.argmax(1)
        center = torch.vstack([centers[i, :] for i in clus_pred]).unsqueeze(1)
        out = trans_IDEC.transformer.decode(ys, center, tgt_mask)
        out = out.transpose(0, 1)
        pred = trans_IDEC.transformer.generator(out[:, -1, :])
        pred = pred.reshape([pred.shape[0], 1, pred.shape[1]])
        ys = torch.cat([ys, pred], dim=0)
    
    tgt_noBOSEOS = tgt[1:max_len, sample_n:(sample_n+1), :]
    ys_noBOSEOS = ys[1:, :, :]
    q = torch.vstack(q)
    return ys[1:, :, :], tgt[1:max_len, sample_n:(sample_n+1), :], ys_noBOSEOS, tgt_noBOSEOS, q



def run_batch_cl2(centers, sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC, src, tgt):
    """
        This function runs part of the inference for each sample
    
    """

    max_len = max_lens[sample_n]
    ys = torch.ones(1, 1, n_tasks).fill_(BOS_IDX).to(DEVICE) 
    q = []
    for i in range(max_len-1):
        src_ = src[0:(i+1), sample_n:(sample_n+1), :].to(DEVICE)
        src_mask = torch.zeros((src_.shape[0], src_.shape[0]),device=DEVICE).type(torch.bool) 
        tgt_mask = torch.zeros((ys.shape[0], ys.shape[0]),device=DEVICE).type(torch.bool)
        memory = trans_IDEC.transformer.encode(src_, src_mask).to(DEVICE)
        out = trans_IDEC.transformer.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)

        pred = trans_IDEC.transformer.generator(out[:, -1, :])
        pred = pred.reshape([pred.shape[0], 1, pred.shape[1]])
        tmp_q = trans_IDEC(pred.squeeze(1))
        q += [tmp_q.cpu()]
        clus_pred = tmp_q.argmax(1)
        center = torch.vstack([centers[i, :] for i in clus_pred]).unsqueeze(1)
        ys = torch.cat([ys, pred], dim=0)
    
    tgt_noBOSEOS = tgt[1:max_len, sample_n:(sample_n+1), :]
    ys_noBOSEOS = ys[1:, :, :]
    q = torch.vstack(q)
    return ys[1:, :, :], tgt[1:max_len, sample_n:(sample_n+1), :], ys_noBOSEOS, tgt_noBOSEOS, q


def run_batch2(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, chances, cutoff):
    """
    Performing sinle sample inference with a transformer, selectively using teacher forcing based on chance threshold.
    
    """

    max_len = max_lens[sample_n]
    ys = torch.ones(1, 1, n_tasks).fill_(BOS_IDX).to(DEVICE) #.type(torch.long)

    for i in range(max_len-1):
        chance = chances[i, sample_n]
        if chance > cutoff:
            tgt_fill = tgt[(i+1), sample_n:(sample_n+1), :]
            tgt_fill = tgt_fill.reshape([tgt_fill.shape[0], 1, tgt_fill.shape[1]])
            ys = torch.cat([ys, tgt_fill], dim=0)

        else:
            src_ = src[0:(i+1), sample_n:(sample_n+1), :]
            
            src_mask = torch.zeros((src_.shape[0], src_.shape[0]),device=DEVICE).type(torch.bool) 
            tgt_mask = torch.zeros((ys.shape[0], ys.shape[0]),device=DEVICE).type(torch.bool)
            memory = transformer.encode(src_, src_mask).to(DEVICE)
            out = transformer.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)

            pred = transformer.generator(out[:, -1, :])
            pred = pred.reshape([pred.shape[0], 1, pred.shape[1]])
            ys = torch.cat([ys, pred], dim=0)

    tgt_noBOSEOS = tgt[1:max_len, sample_n:(sample_n+1), :]
    ys_noBOSEOS = ys[1:, :, :]
    
    return (
        ys[1:, :, :],
        tgt[1:max_len,sample_n:(sample_n+1), :],
        ys_noBOSEOS,
        tgt_noBOSEOS
        )


def run_batch3(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, cutoff):
    """ 
    Performs  single simple inference with a transformer dynamically using teacher forcing
    based on the distance between the model's prediction and the target  value.
    
    """

    max_len = max_lens[sample_n]
    ys = torch.ones(1, 1, n_tasks).fill_(BOS_IDX).to(DEVICE)
    intervene = []
    
    for i in range(max_len-1):
        src_ = src[0:(i+1), sample_n:(sample_n+1), :]
        src_mask = torch.zeros((src_.shape[0], src_.shape[0]),device=DEVICE).type(torch.bool) 
        tgt_mask = torch.zeros((ys.shape[0], ys.shape[0]),device=DEVICE).type(torch.bool) 
        memory = transformer.encode(src_, src_mask).to(DEVICE)
        out = transformer.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        pred = transformer.generator(out[:, -1, :])
        pred = pred.reshape([pred.shape[0], 1, pred.shape[1]])
        tgt_fill = tgt[(i+1), sample_n:(sample_n+1), :]
        tgt_fill = tgt_fill.reshape([tgt_fill.shape[0], 1, tgt_fill.shape[1]])
        dist = math.dist(pred.detach().cpu().flatten().numpy(), tgt_fill.detach().cpu().flatten().numpy())

        if dist > cutoff:
            ys = torch.cat([ys, tgt_fill], dim=0)
            intervene += [1]
        else: 
            ys = torch.cat([ys, pred], dim=0)
            intervene += [0]
        
    tgt_noBOSEOS = tgt[1:max_len, sample_n:(sample_n+1), :]
    ys_noBOSEOS = ys[1:, :, :]
    
    return ys[1:, :, :], tgt[1:max_len, sample_n:(sample_n+1), :], ys_noBOSEOS, tgt_noBOSEOS, intervene


def train_autoRegressive(transformer, n_tasks, train_dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, fraction):
    """
        Trains an transformer in an autoregressive manner, selectively applying teacher forcing
        to entire sequences based on a predefine probability (fraction).

    """

    transformer.train()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    count = 0
    chances = np.random.choice([0, 1], size=(len(train_dataloader)), p=[1-fraction, fraction])

    if np.all(chances == 0): 
        chances[np.random.randint(0, len(chances))] = 1

    for i, (src, tgt) in enumerate(train_dataloader):    
        if chances[i]==1:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            yss = []
            tgts = []
            res = []
            count += 1

            for sample_n in range(src.shape[1]):
                res += [run_batch(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt)]
            
            yss = [res[i][0] for i in range(len(res))]
            tgts = [res[i][1] for i in range(len(res))]
            y_all += [res[i][2].reshape(-1) for i in range(len(res))]
            tgt_all += [res[i][3].reshape(-1) for i in range(len(res))]
            memories += [res[i][4].squeeze(1) for i in range(len(res))]
            yss = torch.concat(yss, axis=0).to(DEVICE)
            tgts = torch.concat(tgts, axis=0).to(DEVICE)
            optimizer.zero_grad()
            
            loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            n_samples += src.shape[1]
            torch.cuda.empty_cache()
            gc.collect()
    
    return losses/count, y_all, tgt_all, n_samples, memories



def train_autoRegressive_sl(transformer, n_tasks, train_dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, fraction, sl=5):
    """
    This function trains a transformer model in an autoregressive manner. 
    """

    transformer.train()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    count = 0
    chances = np.random.choice([0, 1], size=(len(train_dataloader)), p=[1-fraction, fraction])

    if np.all(chances == 0): 
        chances[np.random.randint(0, len(chances))] = 1
    
    for i, (src, tgt) in enumerate(train_dataloader):    
        if chances[i]==1:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            yss = []
            tgts = []
            res = []
            count += 1

            for sample_n in range(src.shape[1]):
                res += [run_batch_sliding(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, sl)]
            
            yss = [res[i][0] for i in range(len(res))]
            tgts = [res[i][1] for i in range(len(res))]
            y_all += [res[i][2].reshape(-1) for i in range(len(res))]
            tgt_all += [res[i][3].reshape(-1) for i in range(len(res))]
            memories += [res[i][4].squeeze(1) for i in range(len(res))]
            yss = torch.concat(yss, axis=0).to(DEVICE)
            tgts = torch.concat(tgts, axis=0).to(DEVICE)
            optimizer.zero_grad()

            loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            n_samples += src.shape[1]

            torch.cuda.empty_cache()
            gc.collect()
    
    return losses/count, y_all, tgt_all, n_samples, memories


def train_autoRegressive_clus(centers, p, trans_IDEC, n_tasks, train_dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE):

    trans_IDEC.train()
    trans_IDEC.transformer.train()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    count = 0
    n_samples_last = 0
    last_q = 0

    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
        yss = []
        tgts = []
        res = []
        count += 1
        n_samples += src.shape[1]

        for sample_n in range(src.shape[1]):
            res += [run_batch_cl(centers, sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC, src, tgt)]
                        
        yss = [res[i][0] for i in range(len(res))]
        tgts = [res[i][1] for i in range(len(res))]
        y_all += [res[i][2].detach().cpu().numpy().reshape(-1) for i in range(len(res))]
        tgt_all += [res[i][3].detach().cpu().numpy().reshape(-1) for i in range(len(res))]
        q = torch.cat([res[i][4] for i in range(len(res))], axis=0)
        yss = torch.concat(yss, axis=0).to(DEVICE)
        tgts = torch.concat(tgts, axis=0).to(DEVICE)
        optimizer.zero_grad()

        loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2).cpu() + F.kl_div(q.log(), p[last_q:(last_q+q.shape[0]), :])
        loss = loss.to(DEVICE)
        loss.backward(retain_graph=True)
        optimizer.step()
        losses += loss.item()

        torch.cuda.empty_cache()
        gc.collect()

        n_samples_last = n_samples
        last_q += q.shape[0]
                
    return losses/count, y_all, tgt_all, n_samples




def train_autoRegressive_q2(centers, p, trans_IDEC, n_tasks, train_dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE):
    """
    Train the auto_regressive model in a multi-task learning setting.
    """

    trans_IDEC.train()
    trans_IDEC.transformer.train()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    count = 0
    n_samples_last = 0
    last_q = 0
    tmp_q = []
    memories = []

    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
        yss = []
        tgts = []
        res = []
        count += 1
        n_samples += src.shape[1]
        for sample_n in range(src.shape[1]):
            res += [run_batch(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC.transformer, src, tgt)]
        
        yss = [res[i][0] for i in range(len(res))]
        tgts = [res[i][1] for i in range(len(res))]
        y_all_ = [res[i][2].reshape(-1) for i in range(len(res))]
        y_all += y_all_
        tgt_all += [res[i][3].reshape(-1) for i in range(len(res))]
        mem = [res[i][4].squeeze(1) for i in range(len(res))]
        memories += mem
        yss = torch.concat(yss, axis=0).to(DEVICE)
        tgts = torch.concat(tgts, axis=0).to(DEVICE)
        z = torch.Tensor(np.concatenate(y_all_).reshape(-1, 17)).to(DEVICE)
        q = trans_IDEC(z).detach().cpu()
        tmp_q += [q]
        optimizer.zero_grad()

        loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2).cpu() + F.kl_div(q.log(), p[last_q:(last_q+q.shape[0]), :])
        loss = loss.to(DEVICE)
        loss.backward(retain_graph=True)
        optimizer.step()
        losses += loss.item()

        torch.cuda.empty_cache()
        gc.collect()

        n_samples_last = n_samples
        last_q += q.shape[0]
                
    return losses/count, y_all, tgt_all, n_samples, tmp_q


def inference_clus2(centers, p, trans_IDEC, n_tasks, train_dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE):
    """
    Performs inference with the trained auto_regressive model in a multi-task learning setting using clustering
    """

    trans_IDEC.transformer.eval()
    trans_IDEC.eval()
    losses = 0
    n_samples = 0
    n_samples_last = 0
    y_all = []
    tgt_all = []
    count = 0
    last_q = 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(train_dataloader):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            yss = []
            tgts = []
            res = []
            count += 1
            n_samples += src.shape[1]

            for sample_n in range(src.shape[1]):
                res += [run_batch_cl2(centers, sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC, src, tgt)]
            
            yss = [res[i][0] for i in range(len(res))]
            tgts = [res[i][1] for i in range(len(res))]
            y_all += [res[i][2].detach().cpu().numpy().reshape(-1) for i in range(len(res))]
            tgt_all += [res[i][3].detach().cpu().numpy().reshape(-1) for i in range(len(res))]
            q = torch.cat([res[i][4] for i in range(len(res))], axis=0)
            yss = torch.concat(yss, axis=0).to(DEVICE)
            tgts = torch.concat(tgts, axis=0).to(DEVICE)
            loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2) + F.kl_div(q.log(), p[last_q:(last_q+q.shape[0]), :])
            losses += loss.item()
            n_samples_last = n_samples

            torch.cuda.empty_cache()
            gc.collect()

            last_q += q.shape[0]
            
    return losses/count, y_all, tgt_all, n_samples


def inference_sample_sl(transformer, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, every, sl=5):
    """
    Performs inference on a subset of the data using a sliding window approach.
    """

    transformer.eval()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    count = 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):    
            if i % every == 0:
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
                yss = []
                tgts = []
                res = []
                count += 1

                for sample_n in range(src.shape[1]):
                    res += [run_batch_sliding(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, sl)]
                
                yss = [res[i][0] for i in range(len(res))]
                tgts = [res[i][1] for i in range(len(res))]
                y_all += [res[i][2].reshape(-1) for i in range(len(res))]
                tgt_all += [res[i][3].reshape(-1) for i in range(len(res))]
                memories += [res[i][4].squeeze(1) for i in range(len(res))]
                yss = torch.concat(yss, axis=0).to(DEVICE)
                tgts = torch.concat(tgts, axis=0).to(DEVICE)
                loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2)
                losses += loss.item()
                n_samples += src.shape[1]

                torch.cuda.empty_cache()
                gc.collect()
    
    return losses/count, y_all, tgt_all, n_samples, memories



def inference_sample(transformer, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, every=2):
    """
    Performs inferene on a subset of the data.

    Iterates through the dataloader, processing only a portion of the data determined ny the (every)
    parameter. 
    """

    transformer.eval()
    losses = 0
    n_samples = 0
    count = 0
    y_all = []
    tgt_all = []
    memories = []

    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):
            if i % every == 0:
                count += 1
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
                yss = []
                tgts = []
                res = []
                torch.manual_seed(count)
                chances = torch.rand(src.shape[0], src.shape[1])

                for sample_n in range(src.shape[1]):
                    res += [run_batch(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt)]
                
                yss = [res[i][0] for i in range(len(res))]
                tgts = [res[i][1] for i in range(len(res))]
                y_all += [res[i][2].reshape(-1) for i in range(len(res))]
                tgt_all += [res[i][3].reshape(-1) for i in range(len(res))]
                memories += [res[i][4].squeeze(1) for i in range(len(res))]
                yss = torch.concat(yss, axis=0).to(DEVICE)
                tgts = torch.concat(tgts, axis=0).to(DEVICE)
                loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2)
                losses += loss.item()
                n_samples += src.shape[1]

                torch.cuda.empty_cache()
                gc.collect()
            
    return losses/count, y_all, tgt_all, n_samples, memories



def inference(transformer, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE):
    """
    Performs inference using the trained transformer model.
    """

    transformer.eval()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    count = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            count += 1
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            yss = []
            tgts = []
            res = []

            for sample_n in range(src.shape[1]):
                res += [run_batch(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt)]
            
            yss = [res[i][0] for i in range(len(res))]
            tgts = [res[i][1] for i in range(len(res))]
            y_all += [res[i][2].reshape(-1) for i in range(len(res))]
            tgt_all += [res[i][3].reshape(-1) for i in range(len(res))]
            memories += [res[i][4].squeeze(1) for i in range(len(res))]
            yss = torch.concat(yss, axis=0).to(DEVICE)
            tgts = torch.concat(tgts, axis=0).to(DEVICE)
            loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2)
            losses += loss.item()
            n_samples += src.shape[1]

            torch.cuda.empty_cache()
            gc.collect()
    
    return losses / len(list(dataloader)), y_all, tgt_all, n_samples, memories


def inference_prescription(transformer, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE, cutoff):
    """
    This function will run inference such that the top 10% furthest prescription vs. AI are replaced with
    the prescription itself.
    """

    transformer.eval()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    intervene = []
    count = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            count += 1
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            yss = []
            tgts = []
            res = []

            for sample_n in range(src.shape[1]):
                res += [run_batch3(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, cutoff)]
            
            yss = [res[i][0] for i in range(len(res))]
            tgts = [res[i][1] for i in range(len(res))]
            intervene += [res[i][4] for i in range(len(res))]
            y_all += [res[i][2].detach().cpu().numpy().reshape(-1) for i in range(len(res))]
            tgt_all += [res[i][3].detach().cpu().numpy().reshape(-1) for i in range(len(res))]
            yss = torch.concat(yss, axis=0).to(DEVICE)
            tgts = torch.concat(tgts, axis=0).to(DEVICE)
            mask = (yss.reshape(-1) == tgts.reshape(-1))
            loss = torch.mean((yss.reshape(-1)[~mask]-tgts.reshape(-1)[~mask])**2)
            losses += loss.item()
            n_samples += src.shape[1]

            torch.cuda.empty_cache()
            gc.collect()
            
    return losses / len(list(dataloader)), y_all, tgt_all, n_samples, intervene


def inference_predict(transformer, n_tasks, dataloader, PAD_IDX, BOS_IDX, DEVICE):
    """
    Perform inference with the trained transformer model and returns predictions and targets.
    """

    transformer.eval()
    y_all = []
    tgt_all = []
    count = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            count += 1
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            res = []
            torch.manual_seed(count)
            chances = torch.rand(src.shape[0], src.shape[1])
            for sample_n in range(src.shape[1]):
                # res += [run_batch2(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt, chances)]
                res += [run_batch(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt)]
            
            y_all += [res[i][2] for i in range(len(res))]
            tgt_all += [res[i][3] for i in range(len(res))]

            torch.cuda.empty_cache()
            gc.collect()
    
    return np.concatenate(y_all, axis=0), np.concatenate(tgt_all, axis=0)
    

def inference_predict_clus(km, transformer, n_tasks, dataloader, PAD_IDX, BOS_IDX, DEVICE):
    """
    Performs inference with the trained transformer model, incorporating clustering for predictions.
    """

    transformer.eval()
    y_all = []
    tgt_all = []
    count = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            count += 1
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            res = []
            torch.manual_seed(count)
            chances = torch.rand(src.shape[0], src.shape[1])

            for sample_n in range(src.shape[1]):
                res += [run_batch_km(km, sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, transformer, src, tgt)]
            
            y_all += [res[i][2].detach().cpu().numpy() for i in range(len(res))]
            tgt_all += [res[i][3].detach().cpu().numpy() for i in range(len(res))]

            torch.cuda.empty_cache()
            gc.collect()
    
    return np.concatenate(y_all, axis=0), np.concatenate(tgt_all, axis=0)


def run_encode(sample_n, max_lens, DEVICE, transformer, src):

    """
    This function runs part of the inference for each sample.
    """

    max_len = max_lens[sample_n]
    memories = []

    for i in range(max_len-1):
        src_ = src[0:(i+1), sample_n:(sample_n+1), :]
        src_mask = torch.zeros((src_.shape[0], src_.shape[0]),device=DEVICE).type(torch.bool)
        memories += [transformer.encode(src_, src_mask).detach().to('cpu')[-1, :, :]]
    
    return torch.cat(memories, axis=0)


def inference_encode(transformer, dataloader, PAD_IDX, DEVICE):
    """
    Performs inference with the trained transformer model, using clustering for predictions.
    """

    transformer.eval()
    memories = []

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            res = []

            for sample_n in range(src.shape[1]):
                res += [run_encode(sample_n, max_lens, DEVICE, transformer, src)]      
            memories += [res[i] for i in range(len(res))]

            torch.cuda.empty_cache()
            gc.collect()
    
    return torch.cat(memories, axis=0)



def inference_q(trans_IDEC, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE):
    """
    Performs infernece with thetrained model and obtain task representation (q).
    """

    trans_IDEC.transformer.eval()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    tmp_q = []
    count = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            count += 1
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            yss = []
            tgts = []
            res = []

            for sample_n in range(src.shape[1]):
                res += [run_batch(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC.transformer, src, tgt)]
            
            yss = [res[i][0] for i in range(len(res))]
            tgts = [res[i][1] for i in range(len(res))]
            y_all += [res[i][2].reshape(-1) for i in range(len(res))]
            tgt_all += [res[i][3].reshape(-1) for i in range(len(res))]
            mem = [res[i][4].squeeze(1) for i in range(len(res))]
            memories += mem
            z = torch.Tensor(np.concatenate(mem)).to(DEVICE)
            tmp_q += [trans_IDEC(z).detach().cpu()]
            yss = torch.concat(yss, axis=0).to(DEVICE)
            tgts = torch.concat(tgts, axis=0).to(DEVICE)
            loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2)
            losses += loss.item()
            n_samples += src.shape[1]

            torch.cuda.empty_cache()
            gc.collect()
    
    return losses / len(list(dataloader)), y_all, tgt_all, n_samples, memories, tmp_q



def inference_q2(trans_IDEC, n_tasks, dataloader, optimizer, PAD_IDX, BOS_IDX, DEVICE):
    """
    Performs inference and calculates tasks representations (q) after preprocessing 
    all the data.
    """

    trans_IDEC.transformer.eval()
    losses = 0
    n_samples = 0
    y_all = []
    tgt_all = []
    memories = []
    tmp_q = []
    count = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            count += 1
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            max_lens = (src != PAD_IDX).transpose(0, 1)[:, :, 0].sum(axis=1)
            yss = []
            tgts = []
            res = []

            for sample_n in range(src.shape[1]):
                res += [run_batch(sample_n, n_tasks, max_lens, BOS_IDX, DEVICE, trans_IDEC.transformer, src, tgt)]
            
            yss = [res[i][0] for i in range(len(res))]
            tgts = [res[i][1] for i in range(len(res))]
            y_all += [res[i][2].reshape(-1) for i in range(len(res))]
            tgt_all += [res[i][3].reshape(-1) for i in range(len(res))]
            mem = [res[i][4].squeeze(1) for i in range(len(res))]
            memories += mem
            yss = torch.concat(yss, axis=0).to(DEVICE)
            tgts = torch.concat(tgts, axis=0).to(DEVICE)
            loss = torch.mean((yss.reshape(-1)-tgts.reshape(-1))**2)
            losses += loss.item()
            n_samples += src.shape[1]
            
            torch.cuda.empty_cache()
            gc.collect()
    
    z = torch.Tensor(np.concatenate(y_all).reshape(-1, 17)).to(DEVICE)
    tmp_q += [trans_IDEC(z).detach().cpu()]
    
    return losses / len(list(dataloader)), y_all, tgt_all, n_samples, memories, tmp_q
