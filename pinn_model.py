"""
Macro‑Informed Neural Network with economic factor constraint.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class EconomicNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def factor_loss(pred, target, anchor_idx=None, lambda_factor=0.2):
    mse = nn.MSELoss()(pred, target)
    if anchor_idx is not None:
        # economic constraint: mean prediction ≈ target mean + anchor alignment
        anchor_pred = pred[:, anchor_idx]
        anchor_targ = target[:, anchor_idx]
        constr = nn.MSELoss()(anchor_pred, anchor_targ)
        return mse + lambda_factor * constr
    return mse

def train_model(X, y, tickers, epochs, lr, device, lambda_factor, anchor_ticker=None):
    model = EconomicNet(X.shape[1], y.shape[1], config.HIDDEN_DIMS).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X).to(device)
    y_t = torch.tensor(y).to(device)
    anchor_idx = tickers.index(anchor_ticker) if anchor_ticker in tickers else None
    best_state = None
    best_loss = float('inf')

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_t)
        loss = factor_loss(pred, y_t, anchor_idx, lambda_factor)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep+1) % 50 == 0:
            print(f"    Epoch {ep+1:3d} Loss: {loss.item():.6f}")
    model.load_state_dict(best_state)
    return model

def predict_latest(model, latest_features, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(latest_features).unsqueeze(0).to(device)
        pred = model(x).cpu().numpy().flatten()
    return pred
