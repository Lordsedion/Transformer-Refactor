import torch
from torch.utils.data import DataLoader
import torch
from scipy.stats import pearsonr


def create_variable_length_dataloader(X_data, Y_data, batch_size=16, shuffle=False, PAD_IDX=9999):
    data = list(zip(X_data, Y_data))

    def collate_fn(batch):
        X_batch, Y_batch = zip(*batch)
        X_padded = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=False, padding_value=PAD_IDX)
        Y_padded = torch.nn.utils.rnn.pad_sequence(Y_batch, batch_first=False, padding_value=PAD_IDX)
        return X_padded, Y_padded
    
    shuffle = shuffle and (len(data) > batch_size) # Determines if shuffling is required

    return DataLoader(
        data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
        )


# Creates Padding and Attention masks
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask

    
def create_mask(src, tgt, PAD_IDX, device):
    """
        Creates masks for the source and target sequence
        
    """

    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = generate_square_subsequent_mask(src_seq_len, device) 

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)[:, :, 0]  # <<-- cos it's 3D need to reduce to 2D
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)[:, :, 0]
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def mse_loss(input, target, ignored_indices, reduction):
    mask = torch.isin(target, ignored_indices)
    out = (input[~mask]-target[~mask])**2
    
    if reduction == "mean":
        return torch.mean(out)
    elif reduction == "None":
        return out


def pearsonr_corr(input, target, ignored_indices):
    mask = torch.isin(target, ignored_indices)
    input = input[~mask].detach().cpu().numpy()
    target = target[~mask].detach().cpu().numpy()

    return pearsonr(input, target)[0]


class EarlyStopping:
    """
        Early stopping utility to stop the training when the validation loss stops improving.

    """

    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False


    def __call__(self, val_loss):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss

        elif val_loss > self.best_val_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss didn't improve for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_val_loss = val_loss
            self.counter = 0

        return self.early_stop