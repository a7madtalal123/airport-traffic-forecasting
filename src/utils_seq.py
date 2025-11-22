# src/utils_seq.py
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-6
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

class SeqDS(Dataset):
    def __init__(self, df, seq_len=24, feature_cols=None, target_col='y'):
        self.df = df.sort_values('date_hour')
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.X, self.y = self._build()

    def _build(self):
        Xs, ys = [], []
        g = self.df
        v = g[self.feature_cols].values
        t = g[self.target_col].values
        for i in range(len(g)-self.seq_len):
            Xs.append(v[i:i+self.seq_len])
            ys.append(t[i+self.seq_len])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def chrono_split(df, split_ratio=0.8):
    df = df.sort_values('date_hour').reset_index(drop=True)
    split = int(len(df)*split_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def evaluate(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(sqrt(mean_squared_error(y_true, y_pred)), 3),
        "sMAPE": round(smape(y_true, y_pred), 2)
    }

def train_seq_model(model, train_df, test_df, feature_cols, seq_len=24, epochs=5, lr=1e-3, batch=128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds_tr = SeqDS(train_df, seq_len, feature_cols)
    ds_te = SeqDS(test_df, seq_len, feature_cols)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch*2, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = 1e9
    patience, bad = 3, 0  # Early stopping بسيط
    for ep in range(epochs):
        model.train()
        for Xb, yb in dl_tr:
            Xb = Xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()

        # val
        model.eval()
        vpreds, vtrues = [], []
        with torch.no_grad():
            for Xb, yb in dl_te:
                Xb = Xb.to(device); yb = yb.to(device)
                v = model(Xb)
                vpreds.append(v.detach().cpu().numpy()); vtrues.append(yb.detach().cpu().numpy())
        vy = np.concatenate(vtrues); vp = np.concatenate(vpreds)
        vRMSE = np.sqrt(((vy - vp)**2).mean())

        if vRMSE < best_loss:
            best_loss = vRMSE; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # final preds
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in dl_te:
            Xb = Xb.to(device)
            p = model(Xb)
            preds.append(p.cpu().numpy()); trues.append(yb.numpy())
    y_true = np.concatenate(trues); y_pred = np.concatenate(preds)
    return y_true, y_pred
