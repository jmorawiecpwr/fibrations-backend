from fastapi import APIRouter
from pydantic import BaseModel
import random

router = APIRouter()

class ModelInput(BaseModel):
    input_vector: list[int]
    iterations: int
    epochs: int

@router.post("/run-model")
def run_model(data: ModelInput):
    return {
        "heatmap": [[random.randint(0, 255) for _ in range(10)] for _ in range(10)],
        "loss": [1.0 / (i + 1) for i in range(10)],
        "iteration_id": 1,
        "description": "Mocked output based on input_vector"
    }