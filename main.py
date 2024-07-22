import os
import pandas as pd
import numpy as np
import gc
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import math
import copy
from itertools import compress
from sklearn.cluster import KMeans
from model import *
from train import *
from utils import *
from .vars import input_var, vars_3

# Load the data using pandas
path = "/data/mock_data.csv"
data_test = pd.read_csv(path, low_memory=False)
X_ = data_test.set_index("MedicalRecordNum", drop=False).loc[:, data_test.columns.isin(input_var)].copy()
y_ = data_test.set_index("MedicalRecordNum", drop=False).loc[:, data_test.columns.isin(vars_3)].copy() 
X_list = [torch.Tensor(X_.loc[i, :].to_numpy()) for i in X_.index.unique()]
y_list = [torch.Tensor(y_.loc[i, :].to_numpy()) for i in y_.index.unique()]
X_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in X_list] 
y_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in y_list]


def create_random_masks(data_length, val_portion, test_portion, random_state=42):

    np.random.seed(random_state)
    num_val = int(data_length * val_portion)
    num_test = int(data_length * test_portion)
    indices = np.random.permutation(data_length)

    # Initialize masks for training, validation and test sets
    mask_train = np.ones(data_length, dtype=bool)
    mask_val = np.zeros(data_length, dtype=bool)
    mask_test = np.zeros(data_length, dtype=bool)
    mask_val[indices[:num_val]] = True
    mask_test[indices[num_val:num_val + num_test]] = True
    mask_train[mask_val | mask_test] = False

    return mask_train, mask_val, mask_test


mask_train, mask_val, mask_test = create_random_masks(
    data_length=len(X_list),
    val_portion=0.1,
    test_portion=0.5,
    random_state=0
    )

input_scaler = StandardScaler()
input_scaler.fit(np.concatenate([i.unsqueeze(0) if len(i.shape)==1 else i for i in list(compress(X_list, mask_train))]))
output_scaler = StandardScaler()
output_scaler.fit(np.concatenate([i.unsqueeze(0) if len(i.shape)==1 else i for i in list(compress(y_list, mask_train))]))

X_list = [torch.Tensor(input_scaler.transform(i)) for i in X_list]
y_list = [torch.Tensor(output_scaler.transform(i)) for i in y_list]

# Define special symbols and their indices
PAD_IDX, BOS_IDX, EOS_IDX = 9999, 2, 3
y_SOS = torch.Tensor([BOS_IDX]).repeat(y_list[0].shape[1]).unsqueeze(0)
y_EOS = torch.Tensor([EOS_IDX]).repeat(y_list[0].shape[1]).unsqueeze(0)
X_EOS = torch.Tensor([EOS_IDX]).repeat(X_list[0].shape[1]).unsqueeze(0)
X_list = [torch.vstack((x, X_EOS)) for x in X_list]
y_list = [torch.vstack((y_SOS, y, y_EOS)) for y in y_list]


# Define model and parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
TGT_N_TASKS = y_.shape[1]
SRC_SIZE = X_.shape[1]
TGT_SIZE = y_.shape[1]
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 256
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    EMB_SIZE,
    NHEAD,
    TGT_N_TASKS,
    SRC_SIZE,
    TGT_SIZE,
    FFN_HID_DIM
    )

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
optimizer = torch.optim.Adam(
    transformer.parameters(),
    lr=0.0001,
    betas=(0.9, 0.98),
    eps=1e-9
    )

# <<<----- Pretraining Section ----->>>

best_val_loss = 9999
NUM_EPOCHS = 10000
early_stopping = EarlyStopping(
    patience=100,
    delta=0.001,
    verbose=True
    )

train_dataloader = create_variable_length_dataloader(
    list(compress(X_list, mask_train)),
    list(compress(y_list, mask_train)),
    batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX
    )

val_dataloader = create_variable_length_dataloader(
    list(compress(X_list, mask_val)),
    list(compress(y_list, mask_val)),
    batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX
    )

test_dataloader = create_variable_length_dataloader(
    list(compress(X_list, mask_test)),
    list(compress(y_list, mask_test)),
    batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX
    )

for epoch in range(1, NUM_EPOCHS+1):
    train_loss, r_train = train_teacherEnforce(
        transformer,
        optimizer,
        train_dataloader,
        PAD_IDX,
        BOS_IDX,
        EOS_IDX,
        DEVICE
        )
    
    val_loss, r_val = evaluate(
        transformer,
        val_dataloader,
        PAD_IDX,
        BOS_IDX,
        EOS_IDX,
        DEVICE
        )
    
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f},\nTrain r: {r_train:.3f}, Val r: {r_val:.3f}"))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = transformer.state_dict()
        
    if early_stopping(val_loss):
        print("Early stopping!")
        break

transformer.load_state_dict(best_model_state)

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()


# Training 2
losses = 0
r = 0
n_samples = 0
y_all = []
tgt_all = []
NUM_EPOCHS = 5000

early_stopping = EarlyStopping(
    patience=15,
    delta=0.0001,
    verbose=True
    )

for epoch in range(1, NUM_EPOCHS+1):

    train_loss, y_all_, tgt_all_, n_samples_, _ = train_autoRegressive(
        transformer,
        y_.shape[1],
        train_dataloader,
        optimizer,
        PAD_IDX,
        BOS_IDX,
        DEVICE,
        fraction=0.5
        )
    
    val_loss, y_all_val_, tgt_all_val_, n_samples_, _ = inference(
        transformer,
        y_.shape[1],
        val_dataloader,
        optimizer,
        PAD_IDX,
        BOS_IDX,
        DEVICE
        )
    
    test_loss, y_all_test_, tgt_all_test_, _, _ = inference_sample(
        transformer,
        y_.shape[1],
        test_dataloader,
        optimizer,
        PAD_IDX,
        BOS_IDX,
        DEVICE
        )
    
    n_samples_ += n_samples_
    r_train = pearsonr(np.concatenate(y_all_), np.concatenate(tgt_all_))[0]
    r_val = pearsonr(np.concatenate(y_all_val_), np.concatenate(tgt_all_val_))[0]
    r_test = pearsonr(np.concatenate(y_all_test_), np.concatenate(tgt_all_test_))[0]

    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Test loss: {test_loss:.3f},\nTrain r: {r_train:.3f}, Val r: {r_val:.3f}, Test r: {r_test:.3f}"))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = transformer.state_dict()

    if early_stopping(val_loss):
        print("Early stopping!")
        break

# Loading the best saved transformer state
transformer.load_state_dict(best_model_state)
torch.save(transformer.state_dict(), '/state_dicts/cv34_embed512_final.pth')
transformer.load_state_dict(torch.load('/state_dicts/cv34_embed512_final.pth'))

# test set eval
val_loss, y_all_val_, tgt_all_val_, n_samples_, _ = inference(
    transformer,
    y_.shape[1],
    test_dataloader,
    optimizer,
    PAD_IDX,
    BOS_IDX,
    DEVICE
    )

r_test = pearsonr(np.concatenate(y_all_val_), np.concatenate(tgt_all_val_))[0]
print((f"Test loss: {val_loss:.3f}, Test r: {r_test:.3f}"))

# Compute the median of Pearson coreelation coefficients for each variable
np.median([pearsonr(
    np.concatenate(y_all_val_).reshape(-1, 17)[:, i],
    np.concatenate(tgt_all_val_).reshape(-1, 17)[:, i])[0] for i in range(len(vars_3))
    ])

# Make predictions
sample_wise_val, y_val = inference_predict(transformer, y_.shape[1], val_dataloader, PAD_IDX, BOS_IDX, DEVICE)
sample_wise_test, y_test = inference_predict(transformer, y_.shape[1], test_dataloader, PAD_IDX, BOS_IDX, DEVICE)
sample_wise_train, y_train = inference_predict(transformer, y_.shape[1], train_dataloader, PAD_IDX, BOS_IDX, DEVICE)
sample_wise_val_latent = inference_encode(transformer, val_dataloader, PAD_IDX, DEVICE).numpy()
sample_wise_test_latent = inference_encode(transformer, test_dataloader, PAD_IDX, DEVICE).numpy()
sample_wise_train_latent = inference_encode(transformer, train_dataloader, PAD_IDX, DEVICE).numpy()

index_reset = ['MedicalRecordNum', 'OrderNum']

test_data = pd.concat([
    data_test.loc[data_test.index.isin(list(compress(y_.index.unique(), mask_test))), :].reset_index()[index_reset],
    pd.DataFrame(y_test.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
    pd.DataFrame(sample_wise_test_latent, columns=['latent_'+str(i) for i in range(EMB_SIZE)]),
    pd.DataFrame(sample_wise_test.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])
    ], axis=1)


train_data = pd.concat([
    data_test.loc[data_test.index.isin(list(compress(y_.index.unique(), mask_train))), :].reset_index()[index_reset],
    pd.DataFrame(y_train.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
    pd.DataFrame(sample_wise_train_latent, columns=['latent_'+str(i) for i in range(EMB_SIZE)]),
    pd.DataFrame(sample_wise_train.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])],
    axis=1
    )


val_data = pd.concat([
    data_test.loc[data_test.index.isin(list(compress(y_.index.unique(), mask_val))), :].reset_index()[index_reset],
    pd.DataFrame(y_val.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
    pd.DataFrame(sample_wise_val_latent, columns=['latent_'+str(i) for i in range(EMB_SIZE)]),
    pd.DataFrame(sample_wise_val.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])],
    axis=1
    )


# Concatenate train, validation, and test data; reset index and drop duplicates
all_data = pd.concat([train_data, val_data, test_data], axis=0).reset_index(drop=True).drop_duplicates(index_reset, keep='last')
data_test_nodup = data_test.drop_duplicates(index_reset, keep='last')
pre_final_data = data_test_nodup.merge(all_data, on=index_reset)
pre_final_data.to_csv('outputs/df_from_model_test34.csv', index=False)


# <<<----- Improvements ----->>>
# Improvement when we had more prescriptions

aa = np.concatenate(sample_wise_train)
bb = np.concatenate(y_train)
diff = [math.dist(aa[i,:], bb[i,:]) for i in range(aa.shape[0])]

percentiles = [5, 10, 20, 50, 80, 90, 95, 100]
r_prescrip = {p:0 for p in percentiles}
r_prescrip_byNutri = {p:0 for p in percentiles}

for p in percentiles:
    print(f"P: {p}")
    cutoff = np.percentile(diff, p)
    val_loss, y_all_val_, tgt_all_val_, n_samples_, intervene = inference_prescription(
        transformer,
        y_.shape[1],
        test_dataloader,
        optimizer,
        PAD_IDX,
        BOS_IDX,
        DEVICE,
        cutoff=cutoff
        )
    
    intervene = [item for sublist in intervene for item in sublist]
    intervene = (np.hstack([np.repeat(i, len(vars_3)) for i in intervene]))
    mask = intervene==1
    r_prescrip[p] = pearsonr(
        np.concatenate(tgt_all_val_)[~mask],
        np.concatenate(y_all_val_)[~mask]
        )[0]
    
    print(r_prescrip[p])

    r_prescrip_byNutri[p] = [
        pearsonr(
        np.concatenate(tgt_all_val_)[~mask].reshape(-1, 17)[:, i],
        np.concatenate(y_all_val_)[~mask].reshape(-1, 17)[:, i]
        )[0] for i in range(len(vars_3))
        ]

np.concatenate(tgt_all_val_)[~mask] == np.concatenate(y_all_val_)[~mask]
{r:np.nanmedian(r_prescrip_byNutri[r]) for r in r_prescrip_byNutri.keys()}
sample_wise_val_sub, sample_wise_test_sub, sample_wise_train_sub = [], [], []
y_val_sub, y_test_sub, y_train_sub = [], [], []

# Exporting predictions
pd.DataFrame(r_prescrip_byNutri, index=vars_3).loc[:, percentiles].to_csv('performance/transformer_prescription.csv')

percentiles = [90]
for percentile in percentiles:
    cutoff = np.percentile(diff, percentile)

    train_loss, y_all_train_, tgt_all_train_, n_samples_, intervene_train = inference_prescription(
        transformer,
        y_.shape[1],
        train_dataloader,
        optimizer,
        PAD_IDX,
        BOS_IDX,
        DEVICE,
        cutoff=cutoff
        )
    
    val_loss, y_all_val_, tgt_all_val_, n_samples_, intervene_val = inference_prescription(
        transformer,
        y_.shape[1],
        val_dataloader,
        optimizer,
        PAD_IDX,
        BOS_IDX,
        DEVICE,
        cutoff=cutoff
        )
    
    test_loss, y_all_test_, tgt_all_test_, n_samples_, intervene_test = inference_prescription(
        transformer,
        y_.shape[1],
        test_dataloader,
        optimizer,
        PAD_IDX,
        BOS_IDX,
        DEVICE,
        cutoff=cutoff
        )

    # Mask for test data
    mask = (np.concatenate(y_all_test_) == np.concatenate(tgt_all_test_))
    pred = np.concatenate(y_all_test_)
    pred[mask] = np.nan
    sample_wise_test_pre = pred.reshape(-1, 17)

    # Flatten and repeat intervene_train list
    intervene_train = [item for sublist in intervene_train for item in sublist]
    intervene_train = (np.hstack([np.repeat(i, len(vars_3)) for i in intervene_train]))

    # Mask for validation data
    mask = (np.concatenate(y_all_val_) == np.concatenate(tgt_all_val_))
    pred = np.concatenate(y_all_val_)
    pred[mask] = np.nan
    sample_wise_val_pre = pred.reshape(-1, 17)

    # Flatten and repeat intervene_val list
    intervene_val = [item for sublist in intervene_val for item in sublist]
    intervene_val = (np.hstack([np.repeat(i, len(vars_3)) for i in intervene_val]))

    # Mask for train data
    mask = (np.concatenate(y_all_train_) == np.concatenate(tgt_all_train_))
    pred = np.concatenate(y_all_train_)
    pred[mask] = np.nan
    sample_wise_train_pre = pred.reshape(-1, 17)

    # Flatten and repeat intervene_test list
    intervene_test = [item for sublist in intervene_test for item in sublist]
    intervene_test = (np.hstack([np.repeat(i, len(vars_3)) for i in intervene_test]))

    test_data = pd.concat([
        data_test.loc[data_test.MedicalRecordNum.isin(list(compress(y_.index.unique(), mask_test))), :].reset_index()[index_reset],
        pd.DataFrame(y_test.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_test_pre, columns=['presc_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_test.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3]),
        pd.DataFrame(intervene_test.reshape(-1, len(vars_3))[:, 0], columns=['intervene'])],
        axis=1
        )
    
    train_data = pd.concat([
        data_test.loc[data_test.MedicalRecordNum.isin(list(compress(y_.index.unique(), mask_train))), :].reset_index()[index_reset],
        pd.DataFrame(y_train.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_train_pre, columns=['presc_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_train.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3]),
        pd.DataFrame(intervene_train.reshape(-1, len(vars_3))[:, 0], columns=['intervene'])],
        axis=1
        )
    
    val_data = pd.concat([
        data_test.loc[data_test.MedicalRecordNum.isin(list(compress(y_.index.unique(), mask_val))), :].reset_index()[index_reset],
        pd.DataFrame(y_val.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_val_pre, columns=['presc_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_val.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3]),
        pd.DataFrame(intervene_val.reshape(-1, len(vars_3))[:, 0], columns=['intervene'])],
        axis=1
        )

    all_data = pd.concat([train_data, val_data, test_data], axis=0).reset_index(drop=True).drop_duplicates(index_reset, keep='last')
    all_data[['noscaled_scaled_'+str(i) for i in vars_3]] = output_scaler.inverse_transform(all_data[['scaled_'+str(i) for i in vars_3]])
    all_data[['noscaled_pred_'+str(i) for i in vars_3]] = output_scaler.inverse_transform(all_data[['pred_'+str(i) for i in vars_3]])
    all_data[['noscaled_presc_'+str(i) for i in vars_3]] = output_scaler.inverse_transform(all_data[['presc_'+str(i) for i in vars_3]])
    all_data.to_csv(path+'/data/prescrip_pred_data_'+str(100-percentile)+'.csv', index=False)
