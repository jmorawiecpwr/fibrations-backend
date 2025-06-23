from __future__ import annotations
from typing import List, Tuple, NamedTuple
import math
import pathlib
import io
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


class SuggestionWithLoss(NamedTuple):
    suggestion: Suggestion
    loss_history: List[float]


def run_suggestion_logic(
        designs_df: pd.DataFrame,
        weight_set: np.ndarray,
        ref_curves: List[np.ndarray],
        dataset: List[Tuple[np.ndarray, np.ndarray]],
        curve_len: int = 60,
        hidden_dim: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
) -> SuggestionWithLoss:
    designs = designs_df.values.astype(float)
    curves = np.vstack([generate_curve(d, T=curve_len) for d in designs])
    objectives = np.vstack([multi_objective_criteria(c, ref_curves) for c in curves])
    weighted = objectives * weight_set
    pareto = compute_pareto_front(weighted)

    ds_designs = np.vstack([d for _, d in dataset])
    ds_curves = np.vstack([generate_curve(d, T=curve_len) for d in ds_designs])

    X_train = torch.tensor(ds_curves, dtype=torch.float32)
    y_train = torch.tensor(ds_designs, dtype=torch.float32)

    model = SeqRegressor(hidden=hidden_dim, output_dim=18)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_history = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())

    pareto_idxs = np.where(pareto)[0]
    if len(pareto_idxs) == 0:
        best_global = 0
    else:
        X_p = torch.tensor(curves[pareto_idxs], dtype=torch.float32)
        _, std = _mc_dropout_predict(model, X_p, n_samples=25)
        best_local = int(np.argmin(np.linalg.norm(std, axis=1)))
        best_global = pareto_idxs[best_local]

    suggestion = Suggestion(
        pareto_mask=pareto,
        next_design=designs[best_global],
        predicted_curve=curves[best_global],
    )
    return SuggestionWithLoss(suggestion=suggestion, loss_history=loss_history)

app = FastAPI(
    title="Design Suggestion Backend",
    description="Suggests next design point from CSV or vector using multi-objective optimisation and LSTM surrogate.",
    version="3.0.0"  # Zintegrowana wersja
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class CsvPayload(BaseModel):
    weight_set: List[float]


class VectorPayload(BaseModel):
    vector: List[float]
    weight_set: List[float]


class TrainingPayload(VectorPayload):
    epochs: int
    iterations: int


def get_common_data():
    rng = np.random.default_rng(42)
    ref_curves = [rng.random(60) for _ in range(9)]
    dataset = []
    for _ in range(50):
        d = rng.random(18);
        d /= np.linalg.norm(d) + 1e-8
        dataset.append((multi_objective_criteria(generate_curve(d), ref_curves), d))
    return ref_curves, dataset


@app.post("/suggest-next/", response_model=dict)
async def suggest_next_design_from_csv(file: UploadFile = File(...), payload: str = Form(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        df = pd.read_csv(io.BytesIO(await file.read()), header=None)
        if df.shape[1] != 18:
            raise ValueError("CSV must have exactly 18 columns")

        parsed_payload = CsvPayload.parse_raw(payload)
        weight_set = np.array(parsed_payload.weight_set)

        ref_curves, dataset = get_common_data()
        result = run_suggestion_logic(df, weight_set, ref_curves, dataset, epochs=50)

        return {
            "next_design": result.suggestion.next_design.tolist(),
            "predicted_curve": result.suggestion.predicted_curve.tolist(),
            "pareto_mask": result.suggestion.pareto_mask.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/process-vector/", response_model=dict)
async def process_vector(payload: VectorPayload):
    try:
        input_vector = np.array(payload.vector)
        if input_vector.shape != (18,):
            raise ValueError("Input vector must contain exactly 18 elements.")

        df = pd.DataFrame([input_vector])
        weight_set = np.array(payload.weight_set)
        ref_curves, dataset = get_common_data()

        result = run_suggestion_logic(df, weight_set, ref_curves, dataset, epochs=1)

        processed_data = {
            "next_point": result.suggestion.next_design.tolist(),
            "curve": result.suggestion.predicted_curve.tolist(),
            "performance_value": np.sum(result.suggestion.predicted_curve)
        }
        return {
            "input_row_index": 0,
            "original_input_vector": payload.vector,
            "processed_data": processed_data,
            "info": "Wygenerowano z pojedynczego wektora (1 epoka)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred: {str(e)}")


@app.post("/run-training/", response_model=dict)
async def run_training(payload: TrainingPayload):
    """Przetwarza pojedynczy wektor z zadaną liczbą epok treningu."""
    try:
        input_vector = np.array(payload.vector)
        if input_vector.shape != (18,):
            raise ValueError("Input vector must contain exactly 18 elements.")

        df = pd.DataFrame([input_vector])
        weight_set = np.array(payload.weight_set)
        ref_curves, dataset = get_common_data()

        result = run_suggestion_logic(df, weight_set, ref_curves, dataset, epochs=payload.epochs)

        processed_data = {
            "next_point": result.suggestion.next_design.tolist(),
            "curve": result.suggestion.predicted_curve.tolist(),
            "performance_value": np.sum(result.suggestion.predicted_curve)
        }
        final_result = {
            "input_row_index": 0,
            "original_input_vector": payload.vector,
            "processed_data": processed_data,
            "info": f"Wygenerowano po {payload.epochs} epokach treningu."
        }
        return {
            "final_result": final_result,
            "loss_history": result.loss_history,
            "epochs_completed": payload.epochs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred: {str(e)}")