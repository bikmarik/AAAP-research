"""Stock-wise betas and a shared, linear factor portfolio encoder."""

import torch
from torch import nn


class ConditionalAutoencoder(nn.Module):
    def __init__(self, features: int, factors: int = 3, hidden: int = 0):
        super().__init__()
        if features < 1 or not 1 <= factors <= features + 1 or hidden < 0:
            raise ValueError("Invalid feature, factor, or hidden-layer dimensions.")
        self.beta = (nn.Sequential(nn.Linear(features, hidden), nn.ReLU(),
                                   nn.Linear(hidden, factors))
                     if hidden else nn.Linear(features, factors))
        # No additive return alpha: factors remain linear portfolios of returns.
        self.factor = nn.Linear(features + 1, factors, bias=False)

    def forward(self, z: torch.Tensor, managed: torch.Tensor,
                month_index: torch.Tensor) -> torch.Tensor:
        return (self.beta(z) * self.factor(managed)[month_index]).sum(dim=-1)

    def forecast(self, z: torch.Tensor, past_managed: torch.Tensor) -> torch.Tensor:
        """Forecast from historical factor premia, never current realized returns."""
        if len(past_managed) == 0:
            raise ValueError("Forecasts require at least one past factor observation.")
        return self.beta(z) @ self.factor(past_managed).mean(dim=0)


class DirectMLP(nn.Module):
    def __init__(self, features: int, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z).squeeze(-1)
