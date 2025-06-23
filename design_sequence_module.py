# full file will be continued below
from __future__ import annotations
from typing import List, Tuple, NamedTuple
import math
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

_DEFAULT_SEED = 42
rng = np.random.default_rng(_DEFAULT_SEED)
torch.manual_seed(_DEFAULT_SEED)

def generate_curve(design: np.ndarray, T: int = 60) -> np.ndarray:
    design = np.maximum(design, 0.0)
    design = design / (np.linalg.norm(design) + 1e-8)
    D = design.size

    theta = np.ones(3 * D)
    f, phi, A = np.split(theta, 3)

    def _logistic_sequence(T_: int, seed_: float, r_: float = 4.0) -> np.ndarray:
        seq = np.empty(T_, dtype=float)
        seq[0] = seed_
        for k in range(1, T_):
            seq[k] = r_ * seq[k - 1] * (1.0 - seq[k - 1])
        return seq

    seeds = np.linspace(0.1, 0.9, D)
    chaotic = np.vstack([_logistic_sequence(T, s) for s in seeds]) - 0.5

    t = np.linspace(0.0, 1.0, T)
    y = np.zeros(T)
    for i in range(D):
        eff_freq = f[i] + A[i] * chaotic[i]
        y += design[i] * np.sin(2 * math.pi * eff_freq * t + phi[i])
    return y

def multi_objective_criteria(curve: np.ndarray, ref_curves: List[np.ndarray]) -> np.ndarray:
    criteria = [-np.sum(np.exp(-np.abs(curve - ref))) for ref in ref_curves]
    return np.asarray(criteria)

def compute_pareto_front(criteria: np.ndarray) -> np.ndarray:
    n = criteria.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominated = np.all(criteria <= criteria[i], axis=1) & np.any(criteria < criteria[i], axis=1)
        efficient = efficient & ~dominated
        efficient[i] = True
    return efficient

class SeqRegressor(nn.Module):
    def __init__(self, hidden: int = 64, num_layers: int = 1, output_dim: int = 18):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = torch.relu(self.fc(out))
        out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-8)
        return out

def _mc_dropout_predict(model: nn.Module, X: torch.Tensor, n_samples: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X).cpu().numpy())
    arr = np.array(preds)
    return arr.mean(axis=0), arr.std(axis=0)

class Suggestion(NamedTuple):
    pareto_mask: np.ndarray
    next_design: np.ndarray
    predicted_curve: np.ndarray

def suggest_next_design(
    weight_set: np.ndarray,
    csv_path: str | pathlib.Path,
    ref_curves: List[np.ndarray],
    dataset: List[Tuple[np.ndarray, np.ndarray]],
    curve_len: int = 60,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
) -> Suggestion:

    weight_set = np.asarray(weight_set, dtype=float).flatten()
    assert weight_set.shape == (9,), "weight_set must have 9 elements"

    df = pd.read_csv(csv_path, header=None)
    assert df.shape[1] == 18, "CSV must have exactly 18 columns"
    designs = df.values.astype(float)

    curves = np.vstack([generate_curve(d, T=curve_len) for d in designs])
    objectives = np.vstack([multi_objective_criteria(c, ref_curves) for c in curves])
    weighted = objectives * weight_set
    pareto = compute_pareto_front(weighted)

    ds_curves = np.vstack([generate_curve(d, T=curve_len) for _, d in dataset])
    ds_designs = np.vstack([d for _, d in dataset])

    X_train = torch.tensor(ds_curves, dtype=torch.float32)
    y_train = torch.tensor(ds_designs, dtype=torch.float32)

    model = SeqRegressor(hidden=hidden_dim, output_dim=18)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()

    pareto_idxs = np.where(pareto)[0]
    X_p = torch.tensor(curves[pareto_idxs], dtype=torch.float32)
    _, std = _mc_dropout_predict(model, X_p, n_samples=25)
    best_local = int(np.argmin(np.linalg.norm(std, axis=1)))
    best_global = pareto_idxs[best_local]

    return Suggestion(
        pareto_mask=pareto,
        next_design=designs[best_global],
        predicted_curve=curves[best_global],
    )