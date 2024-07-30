import os, yaml, argparse
import pandas as pd
import numpy as np
import torch
import math
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from itertools import compress
from model import *
from utils import *
from train import inference_predict, inference_encode, inference_prescription
from vars import input_var, vars_3


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


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config_path, train="", output="", model_path=""):
    config = load_config(config_path=config_path)

    if train != None:
        config["inference"]["path"] = train

    if output != None:
        config["inference"]["output"] = output

    if model_path != None:
        config["inference"]["model_save_path"]

    data = pd.read_csv(config["inference"]["path"], low_memory=False)
    X_ = data.set_index("MedicalRecordNum", drop=False).loc[:, data.columns.isin(input_var)].copy()
    y_ = data.set_index("MedicalRecordNum", drop=False).loc[:, data.columns.isin(vars_3)].copy() 
    X_list = [torch.Tensor(X_.loc[i, :].to_numpy()) for i in X_.index.unique()]
    y_list = [torch.Tensor(y_.loc[i, :].to_numpy()) for i in y_.index.unique()]
    X_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in X_list] 
    y_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in y_list]

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
    DEVICE = config["inference"]["device"]

    BATCH_SIZE = config["inference"]["batch_size"]

    transformer = Seq2SeqTransformer(
        config["model"]["num_encoder_layers"],
        config["model"]["num_decoder_layers"],
        config["model"]["emb_size"],
        config["model"]["nhead"],
        len(vars_3),
        X_.shape[1],
        y_.shape[1],
        config["model"]["ffn_hid_dim"]
    )
    transformer.to(device=DEVICE)

    BATCH_SIZE = 256
    mask_train, mask_val, mask_test = create_random_masks(
        data_length=len(X_list),
        val_portion=0.1,
        test_portion=0.5,
        random_state=0
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

    sample_wise_val, y_val = inference_predict(transformer, y_.shape[1], val_dataloader, PAD_IDX, BOS_IDX, DEVICE)
    sample_wise_test, y_test = inference_predict(transformer, y_.shape[1], test_dataloader, PAD_IDX, BOS_IDX, DEVICE)
    sample_wise_train, y_train = inference_predict(transformer, y_.shape[1], train_dataloader, PAD_IDX, BOS_IDX, DEVICE)
    sample_wise_val_latent = inference_encode(transformer, val_dataloader, PAD_IDX, DEVICE).numpy()
    sample_wise_test_latent = inference_encode(transformer, test_dataloader, PAD_IDX, DEVICE).numpy()
    sample_wise_train_latent = inference_encode(transformer, train_dataloader, PAD_IDX, DEVICE).numpy()

    aa = np.concatenate(sample_wise_train)
    bb = np.concatenate(y_train)
    diff = [math.dist(aa[i,:], bb[i,:]) for i in range(aa.shape[0])]

    percentiles = [5, 10, 20, 50, 80, 90, 95, 100]
    r_prescrip = {p:0 for p in percentiles}
    r_prescrip_byNutri = {p:0 for p in percentiles}
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--config_path", required=True, help="Path to the config file")
    parser.add_argument("--train", required=False, help="Path to training data")
    parser.add_argument("--output", required=False, help="Output directory")
    parser.add_argument("--model_path", required=False, help="Path to the model file")

    args = parser.parse_args()

    main(config_path=args.config_path, train=args.train, output=args.output, model_path=args.model_path)
