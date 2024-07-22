import os
path = "~/home/project_upwork/"
import pandas as pd
import numpy as np
import gc
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import copy
from itertools import compress
from sklearn.cluster import KMeans
from model import *
from train import *
from utils import *

#################################################
input_var = ["gest_age", "bw", "day_since_birth", 'RxDay',
            # 'RxDayShifted', 'shiftedBy',
            "TodaysWeight", "TPNHours", "max_chole_TPNEHR",
            "Alb_lab_value","Ca_lab_value","Cl_lab_value","Glu_lab_value","Na_lab_value", 'BUN_lab_value',
            'Cr_lab_value','Tri_lab_value','ALKP_lab_value','CaI_lab_value',
            'CO2_lab_value', 'PO4_lab_value', 'K_lab_value', 'Mg_lab_value', 'AST_lab_value',
            "ALT_lab_value",
             "EnteralDose", "FluidDose", "VTBI", "InfusionRate",
             "FatInfusionRate",
            "ProtocolName_NEONATAL", "ProtocolName_PEDIATRIC",
            "LineID_1", "LineID_2",
            'gender_concept_id_0', 'gender_concept_id_8507', 'gender_concept_id_8532',  #<<<<<<<<<<<<< proccess it
            'race_concept_id_0', 'race_concept_id_8515', 'race_concept_id_8516',
            'race_concept_id_8527', 'race_concept_id_8557', 'race_concept_id_8657',
            'FatProduct_SMOFlipid 20%', 'FatProduct_Intralipid 20%', 'FatProduct_Omegaven 10%']

# 'DBili_lab_value',
# 'Zn_lab_value',

vars_3 = ['FatDose', 'AADose', 'DexDose', 'Acetate', 'Calcium', 'Copper', 'Famotidine', 'Heparin', 'Levocar',
          'Magnesium', 'MVIDose', 'Phosphate', 'Potassium', 'Ranitidine', 'Selenium', 'Sodium', 'Zinc']

data_test = pd.read_csv(path+'/mock_data.csv', low_memory=False) ##<<< LOAD DATA

# define X and y
# X_ = data_test.loc[:, ~data_test.columns.isin(vars_3)].copy().iloc[:, model.feature_importances_.argsort()[-9:][::-1]]
X_ = data_test.set_index('MedicalRecordNum', drop=False).loc[:, data_test.columns.isin(input_var)].copy()
y_ = data_test.set_index('MedicalRecordNum', drop=False).loc[:, vars_3].copy()
# output_scaler = StandardScaler()
# X_.loc[:, :] = output_scaler.fit_transform(X_)
# output_scaler = StandardScaler()
# y_.loc[:, :] = output_scaler.fit_transform(y_)
X_list = [torch.Tensor(X_.loc[i, :].to_numpy()) for i in X_.index.unique()]
y_list = [torch.Tensor(y_.loc[i, :].to_numpy()) for i in y_.index.unique()]
X_list = [i.unsqueeze(0) if len(i.shape)==1 else i for i in X_list]
y_list = [i.unsqueeze(0) if len(i.shape)==1 else i for i in y_list]
#################################################


################################################
def create_random_masks(data_length, val_portion, test_portion, random_state=42):
    np.random.seed(random_state)
    # Calculate the number of samples for each split
    num_val = int(data_length * val_portion)
    num_test = int(data_length * test_portion)
    # Create a random permutation of indices
    indices = np.random.permutation(data_length)
    # Initialize masks
    mask_train = np.ones(data_length, dtype=bool)
    mask_val = np.zeros(data_length, dtype=bool)
    mask_test = np.zeros(data_length, dtype=bool)
    # Assign samples to each split using the random permutation
    mask_val[indices[:num_val]] = True
    mask_test[indices[num_val:num_val + num_test]] = True
    mask_train[mask_val | mask_test] = False
    return mask_train, mask_val, mask_test

mask_train, mask_val, mask_test = create_random_masks(len(X_list), val_portion=0.1, test_portion=0.5, random_state=0)



input_scaler = StandardScaler()
input_scaler.fit(np.concatenate([i.unsqueeze(0) if len(i.shape)==1 else i for i in list(compress(X_list, mask_train))]))
output_scaler = StandardScaler()
output_scaler.fit(np.concatenate([i.unsqueeze(0) if len(i.shape)==1 else i for i in list(compress(y_list, mask_train))]))

X_list = [torch.Tensor(input_scaler.transform(i)) for i in X_list]
y_list = [torch.Tensor(output_scaler.transform(i)) for i in y_list]
#################################################

#################################################
# padding beginning and ending
# Define special symbols and indices
PAD_IDX, BOS_IDX, EOS_IDX = 9999, 2, 3
y_SOS = torch.Tensor([BOS_IDX]).repeat(y_list[0].shape[1]).unsqueeze(0)
y_EOS = torch.Tensor([EOS_IDX]).repeat(y_list[0].shape[1]).unsqueeze(0)
# X_SOS = torch.Tensor([BOS_IDX]).repeat(X_list[0].shape[1]).unsqueeze(0)
X_EOS = torch.Tensor([EOS_IDX]).repeat(X_list[0].shape[1]).unsqueeze(0)
# X_list = [torch.vstack((X_SOS, x, X_EOS)) for x in X_list]
X_list = [torch.vstack((x, X_EOS)) for x in X_list]
y_list = [torch.vstack((y_SOS, y, y_EOS)) for y in y_list]
#################################################

#################################################
# define model and parameters
DEVICE = "cuda:2" #"cuda:2"
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, TGT_N_TASKS, SRC_SIZE, TGT_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# optimizer = torch.optim.AdamW(transformer.parameters())
#################################################

#################################################
# pretraining
NUM_EPOCHS = 10000
early_stopping = EarlyStopping(patience=100, delta=0.001, verbose=True)

train_dataloader = create_variable_length_dataloader(list(compress(X_list, mask_train)),
                                                     list(compress(y_list, mask_train)),
                                                     batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)
val_dataloader = create_variable_length_dataloader(list(compress(X_list, mask_val)),
                                                   list(compress(y_list, mask_val)),
                                                   batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)
test_dataloader = create_variable_length_dataloader(list(compress(X_list, mask_test)),
                                                list(compress(y_list, mask_test)),
                                                batch_size=BATCH_SIZE, PAD_IDX=PAD_IDX)

best_val_loss = 9999
for epoch in range(1, NUM_EPOCHS+1):
    train_loss, r_train = train_teacherEnforce(transformer, optimizer, train_dataloader, PAD_IDX, BOS_IDX, EOS_IDX, DEVICE)
    val_loss, r_val = evaluate(transformer, val_dataloader, PAD_IDX, BOS_IDX, EOS_IDX, DEVICE)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f},\
             Train r: {r_train:.3f}, Val r: {r_val:.3f}"))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = transformer.state_dict()
    if early_stopping(val_loss):
        print("Early stopping!")
        break

transformer.load_state_dict(best_model_state)
#################################################

# clear GPU
torch.cuda.empty_cache()
gc.collect()


#################################################
# train 2
losses = 0
r = 0
n_samples = 0
y_all = []
tgt_all = []
NUM_EPOCHS = 5000


early_stopping = EarlyStopping(patience=15, delta=0.0001, verbose=True)

for epoch in range(1, NUM_EPOCHS+1):
    train_loss, y_all_, tgt_all_, n_samples_, _ = train_autoRegressive(transformer, y_.shape[1], train_dataloader,
                                                                             optimizer, PAD_IDX, BOS_IDX, DEVICE, fraction=0.5)
    val_loss, y_all_val_, tgt_all_val_, n_samples_, _ = inference(transformer, y_.shape[1], val_dataloader,
                                                                        optimizer, PAD_IDX, BOS_IDX, DEVICE)
    test_loss, y_all_test_, tgt_all_test_, _, _ = inference_sample(transformer, y_.shape[1], test_dataloader,
                                                                        optimizer, PAD_IDX, BOS_IDX, DEVICE)
    # y_all += y_all_
    # tgt_all += tgt_all_
    n_samples_ += n_samples_
    r_train = pearsonr(np.concatenate(y_all_), np.concatenate(tgt_all_))[0]
    r_val = pearsonr(np.concatenate(y_all_val_), np.concatenate(tgt_all_val_))[0]
    r_test = pearsonr(np.concatenate(y_all_test_), np.concatenate(tgt_all_test_))[0]
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Test loss: {test_loss:.3f},\
                Train r: {r_train:.3f}, Val r: {r_val:.3f}, Test r: {r_test:.3f}"))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = transformer.state_dict()
    if early_stopping(val_loss):
        print("Early stopping!")
        break
#################################################

transformer.load_state_dict(best_model_state)

torch.save(transformer.state_dict(), 'state_dicts/cv34_embed512_final.pth')
transformer.load_state_dict(torch.load('state_dicts/cv34_embed512_final.pth'))


#################################################
# test set eval
val_loss, y_all_val_, tgt_all_val_, n_samples_, _ = inference(transformer, y_.shape[1], test_dataloader,
                                                                    optimizer, PAD_IDX, BOS_IDX, DEVICE)
r_test = pearsonr(np.concatenate(y_all_val_), np.concatenate(tgt_all_val_))[0]
print((f"Test loss: {val_loss:.3f}, Test r: {r_test:.3f}"))


np.median([pearsonr(np.concatenate(y_all_val_).reshape(-1, 17)[:, i],
                  np.concatenate(tgt_all_val_).reshape(-1, 17)[:, i])[0] for i in range(len(vars_3))])

#################################################



#################################################
# Make predictions
sample_wise_val, y_val = inference_predict(transformer, y_.shape[1], val_dataloader, PAD_IDX, BOS_IDX, DEVICE)
sample_wise_test, y_test = inference_predict(transformer, y_.shape[1], test_dataloader, PAD_IDX, BOS_IDX, DEVICE)
sample_wise_train, y_train = inference_predict(transformer, y_.shape[1], train_dataloader, PAD_IDX, BOS_IDX, DEVICE)

sample_wise_val_latent = inference_encode(transformer, val_dataloader, PAD_IDX, DEVICE).numpy()
sample_wise_test_latent = inference_encode(transformer, test_dataloader, PAD_IDX, DEVICE).numpy()
sample_wise_train_latent = inference_encode(transformer, train_dataloader, PAD_IDX, DEVICE).numpy()

test_data = pd.concat([data_test.loc[data_test.index.isin(list(compress(y_.index.unique(), mask_test))), :].reset_index()[['MedicalRecordNum', 'OrderNum']],
                       pd.DataFrame(y_test.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
                       pd.DataFrame(sample_wise_test_latent, columns=['latent_'+str(i) for i in range(EMB_SIZE)]),
                       pd.DataFrame(sample_wise_test.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])], axis=1)
# test_data['LineID'] = test_data['LineID_1'].map({0:2, 1:1})
# test_data['ProtocolName'] = test_data['ProtocolName_NEONATAL'].map({0:'PEDIATRIC', 1:'NEONATAL'})
# test_data = test_data.drop([ 'LineID_1', 'ProtocolName_NEONATAL'], axis=1)

train_data = pd.concat([data_test.loc[data_test.index.isin(list(compress(y_.index.unique(), mask_train))), :].reset_index()[['MedicalRecordNum', 'OrderNum']],
                       pd.DataFrame(y_train.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
                       pd.DataFrame(sample_wise_train_latent, columns=['latent_'+str(i) for i in range(EMB_SIZE)]),
                       pd.DataFrame(sample_wise_train.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])], axis=1)
# train_data['LineID'] = train_data['LineID_1'].map({0:2, 1:1})
# train_data['ProtocolName'] = train_data['ProtocolName_NEONATAL'].map({0:'PEDIATRIC', 1:'NEONATAL'})
# train_data = train_data.drop([ 'LineID_1', 'ProtocolName_NEONATAL'], axis=1)

val_data = pd.concat([data_test.loc[data_test.index.isin(list(compress(y_.index.unique(), mask_val))), :].reset_index()[['MedicalRecordNum', 'OrderNum']],
                       pd.DataFrame(y_val.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
                       pd.DataFrame(sample_wise_val_latent, columns=['latent_'+str(i) for i in range(EMB_SIZE)]),
                       pd.DataFrame(sample_wise_val.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])], axis=1)
# val_data['LineID'] = val_data['LineID_1'].map({0:2, 1:1})
# val_data['ProtocolName'] = val_data['ProtocolName_NEONATAL'].map({0:'PEDIATRIC', 1:'NEONATAL'})
# val_data = val_data.drop(['LineID_1', 'ProtocolName_NEONATAL'], axis=1)

all_data = pd.concat([train_data, val_data, test_data], axis=0).reset_index(drop=True).drop_duplicates(['MedicalRecordNum', 'OrderNum'], keep='last')
data_test_nodup = data_test.drop_duplicates(['MedicalRecordNum', 'OrderNum'], keep='last')
pre_final_data = data_test_nodup.merge(all_data, on=['MedicalRecordNum', 'OrderNum'])
pre_final_data.to_csv('outputs/df_from_model_test34.csv', index=False)
#################################################



#################################################
# improvement when we had more prescriptions
import math

aa = np.concatenate(sample_wise_train)
bb = np.concatenate(y_train)
diff = [math.dist(aa[i,:], bb[i,:]) for i in range(aa.shape[0])]
# diff = np.concatenate(sample_wise_train).reshape(-1, 17) - np.concatenate(y_train).reshape(-1, 17)

percentiles = [5, 10, 20, 50, 80, 90, 95, 100]
# percentiles = [20]
r_prescrip = {p:0 for p in percentiles}
r_prescrip_byNutri = {p:0 for p in percentiles}
for p in percentiles:
    print(p)
    cutoff = np.percentile(diff, p)
    # cutoff = np.percentile(np.abs(diff).mean(axis=1), p)
    # cutoff = np.percentile(np.abs(diff.mean(axis=1)), p)
    val_loss, y_all_val_, tgt_all_val_, n_samples_, intervene = inference_prescription(transformer, y_.shape[1], test_dataloader,
                                                                        optimizer, PAD_IDX, BOS_IDX, DEVICE, cutoff=cutoff)
    # r_test = pearsonr(np.concatenate(y_all_val_), np.concatenate(tgt_all_val_))[0]
    intervene = [item for sublist in intervene for item in sublist]
    intervene = (np.hstack([np.repeat(i, len(vars_3)) for i in intervene]))
    # mask = (np.concatenate(tgt_all_val_) == np.concatenate(y_all_val_))
    mask = intervene==1
    r_prescrip[p] = pearsonr(np.concatenate(tgt_all_val_)[~mask], np.concatenate(y_all_val_)[~mask])[0]
    print(r_prescrip[p])
    r_prescrip_byNutri[p] = [pearsonr(np.concatenate(tgt_all_val_)[~mask].reshape(-1, 17)[:, i],
                                      np.concatenate(y_all_val_)[~mask].reshape(-1, 17)[:, i])[0] for i in range(len(vars_3))]
#################################################

np.concatenate(tgt_all_val_)[~mask] == np.concatenate(y_all_val_)[~mask]

{r:np.nanmedian(r_prescrip_byNutri[r]) for r in r_prescrip_byNutri.keys()}

sample_wise_val_sub, sample_wise_test_sub, sample_wise_train_sub = [], [], []
y_val_sub, y_test_sub, y_train_sub = [], [], []


pd.DataFrame(r_prescrip_byNutri, index=vars_3).loc[:, percentiles].to_csv('performance/transformer_prescription.csv')



# exporting predictions
percentiles = [90]

for percentile in percentiles:
    cutoff = np.percentile(diff, percentile)
    train_loss, y_all_train_, tgt_all_train_, n_samples_, intervene_train = inference_prescription(transformer, y_.shape[1], train_dataloader,
                                                                        optimizer, PAD_IDX, BOS_IDX, DEVICE, cutoff=cutoff)
    val_loss, y_all_val_, tgt_all_val_, n_samples_, intervene_val = inference_prescription(transformer, y_.shape[1], val_dataloader,
                                                                        optimizer, PAD_IDX, BOS_IDX, DEVICE, cutoff=cutoff)
    test_loss, y_all_test_, tgt_all_test_, n_samples_, intervene_test = inference_prescription(transformer, y_.shape[1], test_dataloader,
                                                                        optimizer, PAD_IDX, BOS_IDX, DEVICE, cutoff=cutoff)
    #
    mask = (np.concatenate(y_all_test_) == np.concatenate(tgt_all_test_))
    pred = np.concatenate(y_all_test_)
    pred[mask] = np.nan
    sample_wise_test_pre = pred.reshape(-1, 17)
    intervene_train = [item for sublist in intervene_train for item in sublist]
    intervene_train = (np.hstack([np.repeat(i, len(vars_3)) for i in intervene_train]))
    #
    mask = (np.concatenate(y_all_val_) == np.concatenate(tgt_all_val_))
    pred = np.concatenate(y_all_val_)
    pred[mask] = np.nan
    sample_wise_val_pre = pred.reshape(-1, 17)
    intervene_val = [item for sublist in intervene_val for item in sublist]
    intervene_val = (np.hstack([np.repeat(i, len(vars_3)) for i in intervene_val]))
    #
    mask = (np.concatenate(y_all_train_) == np.concatenate(tgt_all_train_))
    pred = np.concatenate(y_all_train_)
    pred[mask] = np.nan
    sample_wise_train_pre = pred.reshape(-1, 17)
    intervene_test = [item for sublist in intervene_test for item in sublist]
    intervene_test = (np.hstack([np.repeat(i, len(vars_3)) for i in intervene_test]))
    #
    test_data = pd.concat([data_test.loc[data_test.MedicalRecordNum.isin(list(compress(y_.index.unique(), mask_test))), :].reset_index()[['MedicalRecordNum', 'OrderNum']],
                        pd.DataFrame(y_test.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
                        pd.DataFrame(sample_wise_test_pre, columns=['presc_'+str(i) for i in vars_3]),
                        pd.DataFrame(sample_wise_test.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3]),
                        pd.DataFrame(intervene_test.reshape(-1, len(vars_3))[:, 0], columns=['intervene'])], axis=1)
    #
    train_data = pd.concat([data_test.loc[data_test.MedicalRecordNum.isin(list(compress(y_.index.unique(), mask_train))), :].reset_index()[['MedicalRecordNum', 'OrderNum']],
                        pd.DataFrame(y_train.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
                        pd.DataFrame(sample_wise_train_pre, columns=['presc_'+str(i) for i in vars_3]),
                        pd.DataFrame(sample_wise_train.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3]),
                        pd.DataFrame(intervene_train.reshape(-1, len(vars_3))[:, 0], columns=['intervene'])], axis=1)
    #
    val_data = pd.concat([data_test.loc[data_test.MedicalRecordNum.isin(list(compress(y_.index.unique(), mask_val))), :].reset_index()[['MedicalRecordNum', 'OrderNum']],
                        pd.DataFrame(y_val.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
                        pd.DataFrame(sample_wise_val_pre, columns=['presc_'+str(i) for i in vars_3]),
                        pd.DataFrame(sample_wise_val.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3]),
                        pd.DataFrame(intervene_val.reshape(-1, len(vars_3))[:, 0], columns=['intervene'])], axis=1)
    #
    all_data = pd.concat([train_data, val_data, test_data], axis=0).reset_index(drop=True).drop_duplicates(['MedicalRecordNum', 'OrderNum'], keep='last')
    all_data[['noscaled_scaled_'+str(i) for i in vars_3]] = output_scaler.inverse_transform(all_data[['scaled_'+str(i) for i in vars_3]])
    all_data[['noscaled_pred_'+str(i) for i in vars_3]] = output_scaler.inverse_transform(all_data[['pred_'+str(i) for i in vars_3]])
    all_data[['noscaled_presc_'+str(i) for i in vars_3]] = output_scaler.inverse_transform(all_data[['presc_'+str(i) for i in vars_3]])
    all_data.to_csv(path+'/data/prescrip_pred_data_'+str(100-percentile)+'.csv', index=False)
