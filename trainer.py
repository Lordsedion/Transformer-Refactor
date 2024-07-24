import yaml
import torch
from model import Seq2SeqTransformer  # Import your model class
from train import train_teacherEnforce, evaluate  # Import your training functions
from utils import create_variable_length_dataloader, EarlyStopping  # Import utility functions
from vars import input_var, vars_3  # Import your variables
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import gc
import torch.nn as nn
from itertools import compress
# from model import 
from train import train_autoRegressive, inference_sample, inference, inference_predict, inference_encode
import pandas as pd
# from main import create_random_masks
from utils import *
import os


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
        config["training"]["path"] = train

    if output != None:
        config["training"]["output"] = output

    if model_path != None:
        config["training"]["model_save_path"]

    data = pd.read_csv(config["training"]["path"], low_memory=False)
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


    # Define model and parameters
    BATCH_SIZE = config["training"]["batch_size"]

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

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    best_val_loss = 9999
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        delta=config["training"]["delta"],
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

    transformer.to(config["training"]["device"])
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config["training"]["lr"], betas=(0.9, 0.98), eps=1e-9)
    early_stopping = EarlyStopping(patience=config["training"]["patience"], delta=config["training"]["delta"], verbose=True)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, config["training"]["num_epochs"]+1):
        train_loss, r_train = train_teacherEnforce(
            model=transformer,
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
        
        print(f"Here: Epoch: {epoch}, Train_loss: {train_loss:.3f} Val loss: {val_loss:.3f} Train r: {r_train:.3f} Val r: {r_val:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = transformer.state_dict()

        if early_stopping(val_loss=val_loss):
            print("Early stopping")
            break
    transformer.load_state_dict(best_model_state)

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # Training 2
    early_stopping = EarlyStopping(
        patience=config["training2"]["patience"],
        delta=config["training"]["delta"],
        verbose=True
        )

    for epoch in range(1, config["training"]["num_epochs"]+1):

        train_loss, y_all_, tgt_all_, n_samples_, _ = train_autoRegressive(
            transformer,
            y_.shape[1],
            train_dataloader,
            optimizer,
            PAD_IDX,
            BOS_IDX,
            config["training"]["device"],
            fraction=0.5
            )
        
        val_loss, y_all_val_, tgt_all_val_, n_samples_, _ = inference(
            transformer,
            y_.shape[1],
            val_dataloader,
            optimizer,
            PAD_IDX,
            BOS_IDX,
            config["training2"]["device"]
            )
        
        test_loss, y_all_test_, tgt_all_test_, _, _ = inference_sample(
            transformer,
            y_.shape[1],
            test_dataloader,
            optimizer,
            PAD_IDX,
            BOS_IDX,
            config["training2"]["device"]
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

    directory_path = os.path.dirname(config["training"]["model_save_path"])
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    torch.save(transformer.state_dict(), config["training"]["model_save_path"])

    # test set eval
    val_loss, y_all_val_, tgt_all_val_, n_samples_, _ = inference(
        transformer,
        y_.shape[1],
        test_dataloader,
        optimizer,
        PAD_IDX,
        BOS_IDX,
        config["training2"]["device"]
        )

    r_test = pearsonr(np.concatenate(y_all_val_), np.concatenate(tgt_all_val_))[0]
    print((f"Test loss: {val_loss:.3f}, Test r: {r_test:.3f}"))

    # Compute the median of Pearson coreelation coefficients for each variable
    np.median([pearsonr(
        np.concatenate(y_all_val_).reshape(-1, 17)[:, i],
        np.concatenate(tgt_all_val_).reshape(-1, 17)[:, i])[0] for i in range(len(vars_3))
        ])

    # Make predictions
    sample_wise_val, y_val = inference_predict(transformer, y_.shape[1], val_dataloader, PAD_IDX, BOS_IDX,  config["training"]["device"])
    sample_wise_test, y_test = inference_predict(transformer, y_.shape[1], test_dataloader, PAD_IDX, BOS_IDX,  config["training"]["device"])
    sample_wise_train, y_train = inference_predict(transformer, y_.shape[1], train_dataloader, PAD_IDX, BOS_IDX,  config["training"]["device"])
    sample_wise_val_latent = inference_encode(transformer, val_dataloader, PAD_IDX,  config["training"]["device"]).numpy()
    sample_wise_test_latent = inference_encode(transformer, test_dataloader, PAD_IDX,  config["training"]["device"]).numpy()
    sample_wise_train_latent = inference_encode(transformer, train_dataloader, PAD_IDX,  config["training"]["device"]).numpy()

    index_reset = ['MedicalRecordNum']

    test_data = pd.concat([
        data.loc[data.index.isin(list(compress(y_.index.unique(), mask_test))), :].reset_index()[index_reset],
        pd.DataFrame(y_test.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_test_latent, columns=['latent_'+str(i) for i in range(config["model"]["emb_size"])]),
        pd.DataFrame(sample_wise_test.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])
        ], axis=1)


    train_data = pd.concat([
        data.loc[data.index.isin(list(compress(y_.index.unique(), mask_train))), :].reset_index()[index_reset],
        pd.DataFrame(y_train.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_train_latent, columns=['latent_'+str(i) for i in range(config["model"]["emb_size"])]),
        pd.DataFrame(sample_wise_train.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])],
        axis=1
        )


    val_data = pd.concat([
        data.loc[data.index.isin(list(compress(y_.index.unique(), mask_val))), :].reset_index()[index_reset],
        pd.DataFrame(y_val.reshape(-1, len(vars_3)), columns=['scaled_'+str(i) for i in vars_3]),
        pd.DataFrame(sample_wise_val_latent, columns=['latent_'+str(i) for i in range(config["model"]["emb_size"])]),
        pd.DataFrame(sample_wise_val.reshape(-1, len(vars_3)), columns=['pred_'+i for i in vars_3])],
        axis=1
        )


    # Concatenate train, validation, and test data; reset index and drop duplicates
    all_data = pd.concat([train_data, val_data, test_data], axis=0).reset_index(drop=True).drop_duplicates(index_reset, keep='last')
    data_test_nodup = data.drop_duplicates(index_reset, keep='last')
    pre_final_data = data_test_nodup.merge(all_data, on=index_reset)

    directory_path = os.path.dirname(config["training"]["output"])
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)

    pre_final_data.to_csv(config["training"]["output"], index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--config_path", required=True, help="Path to the config file")
    parser.add_argument("--train", required=False, help="Path to training data")
    parser.add_argument("--output", required=False, help="Output directory")
    parser.add_argument("--model_path", required=False, help="Path to the model file")

    args = parser.parse_args()

    main(config_path=args.config_path, train=args.train, output=args.output, model_path=args.model_path)



