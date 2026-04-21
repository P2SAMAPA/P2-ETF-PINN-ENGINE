"""
Physics-Informed Neural Network (PINN) with Black-Scholes-Merton PDE constraint.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PINNTrainer:
    def __init__(self, hidden_layers=[64,64,32], epochs=200, batch_size=64, lr=0.001,
                 physics_weight=0.1, seed=42):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.physics_weight = physics_weight
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def pde_loss(self, S, predicted_price, r=0.02, sigma=0.2):
        """
        Black-Scholes-Merton PDE residual: dV/dt + r*S*dV/dS + 0.5*sigma^2*S^2*d2V/dS2 - r*V = 0
        Simplified: We enforce that the predicted return is consistent with a no-arbitrage drift.
        Here we use a simple approximation: the predicted price change should satisfy
        E[dS] = r * S * dt under risk-neutral measure.
        We enforce that the predicted log-return is close to r * dt.
        """
        # S is the input price (scaled), predicted_price is the model output (scaled)
        # We want the drift to be approximately r * dt
        dt = 1/252
        target_drift = r * dt
        pred_drift = predicted_price - S  # difference over one time step
        return torch.mean((pred_drift - target_drift) ** 2)

    def fit(self, prices: np.ndarray):
        """Train PINN on a univariate price series."""
        # Prepare features: lagged prices and time index
        n = len(prices)
        if n < 50:
            return False

        # Create features: (normalized price, time/252)
        scaled_prices = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        X = np.column_stack([
            scaled_prices[:-1],  # S_t
            np.arange(1, n) / 252.0  # time in years
        ])
        y = scaled_prices[1:]  # S_{t+1}

        # Train/validation split
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.model = PINN(input_dim=2, hidden_layers=self.hidden_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_X).squeeze()
                data_loss = mse_loss(pred, batch_y)
                # Physics loss: enforce drift ≈ r * dt
                S_input = batch_X[:, 0]  # scaled price
                phys_loss = self.pde_loss(S_input, pred)
                loss = data_loss + self.physics_weight * phys_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            if (epoch+1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}")

        return True

    def forecast(self, prices: np.ndarray) -> float:
        """Predict next day's return."""
        self.model.eval()
        scaled = self.scaler.transform(prices.reshape(-1, 1)).flatten()
        last_price = scaled[-1]
        t = len(prices) / 252.0
        X = torch.tensor([[last_price, t]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(X).item()
        pred_price = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
        current_price = prices[-1]
        return (pred_price / current_price) - 1.0
