import yaml
import torch
from model import Seq2SeqTransformer  # Import your model class
from train import train_teacherEnforce, evaluate  # Import your training functions
from utils import create_variable_length_dataloader, EarlyStopping  # Import utility functions
from vars import input_var, vars_3  # Import your variables
from sklearn.preprocessing import StandardScaler
import numpy as np
# from model import 
# from train import 
import pandas as pd
from main import create_random_masks
from utils import *
import os


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config_path):
    config = load_config(config_path=config_path)

    # Load data into the memory
    data = pd.read_csv(config["data"]["path"], low_memory=False)
    X_ = data.set_index("MedicalRecordNum", drop=False).loc[:, data.columns.isin(input_var)].copy()
    y_ = data.set_index("MedicalRecordNum", drop=False).loc[:, data.columns.isin(vars_3)].copy()
    X_list = [torch.tensor(X_.loc[i, :].to_numpy()) for i in X_.index.unique()]
    X_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in X_list]
    y_list = [torch.tensor(X_.loc[i, :].to_numpy()) for i in y_.index.unique()] 
    y_list = [i.unsqueeze(0) if len(i.shape) == 1 else i for i in y_list ]

    # Apply data preprocessing here
    input_scaler = StandardScaler()
    input_scaler.fit(np.concatenate([i.unsqueeze(0) if len(i.shape) == 1 else i for i in X_list]))
    output_scaler = StandardScaler()
    output_scaler.fit(np.concatenate([i.unsqueeze(0) if len(i.shape) == 1 else i for i in y_list]))

    X_list = [torch.tensor(input_scaler.transform(i)) for i in X_list]
    y_list = [torch.tensor(output_scaler.transform(i)) for i in y_list]

    # Add Special tokens

    PAD_IDX, BOS_IDX, EOS_IDX = 9999, 2, 3
    y_SOS = torch.tensor([BOS_IDX]).repeat(y_list[0].shape[1]).unsqueeze(0)
    y_EOS = torch.tensor([EOS_IDX]).repeat(y_list[0].shape[1]).unsqueeze(0)
    X_EOS = torch.tensor([EOS_IDX]).repeat(X_list[0].shape[1]).unsqueeze(0)
    X_list = [torch.vstack((x, X_EOS)) for x in X_list]
    y_list = [torch.vstack((y_SOS, y, y_EOS)) for y in y_list]

    mask_train, mask_val, mask_test = create_random_masks(len(X_list), 0.1, 0.5, random_state=0)

    train_dataloader = create_variable_length_dataloader(X_list, y_list, mask_train, config["training"]["batch_size"], PAD_IDX=PAD_IDX)
    val_dataloader = create_variable_length_dataloader(X_list, y_list, mask_val, config["training"]["batch_size"], PAD_IDX=PAD_IDX)

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

    transformer.to(config("training")["device"])
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config["training"]["lr"], betas=(0.9, 0.98), eps=1e-9)
    early_stopping = EarlyStopping(patience=config["training"]["patience"], delta=config["training"]["delta"], verbose=True)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, config["traiing"]["num_epochs"]+1):
        train_loss, r_train = train_teacherEnforce(
            transformer=transformer,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            PAD_IDX=PAD_IDX,
            BOS_IDX=BOS_IDX,
            EOS_IDX=EOS_IDX,
            DEVICE=config["training"]["device"]
             )
        
        val_loss, r_val = evaluate(
         transformer,
         val_dataloader,
         PAD_IDX,
         BOS_IDX,
         EOS_IDX,
         config['training']['device']
         )
        
        print(f"Epoch: {epoch}, Train_loss: {train_loss:.3f} Val loss: {val_loss:.3f} Train r: {r_train:.3f} Val r: {r_val:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = transformer.state_dict()

        if early_stopping(val_loss=val_loss):
            print("Early stopping")
            break
    transformer.load_state_dict(best_model_state)
    if not os.path.isdir(config["training"]["model_save_path"]):
        os.makedirs(config["training"]["model_save_path"])
    torch.save(transformer.state_dict(), config["training"]["model_save_path"])


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1]
    main(config_path=config_path)


    

