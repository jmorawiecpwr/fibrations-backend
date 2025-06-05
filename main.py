from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import csv_mock, model_mock, categories_mock

import csv
import io
import random
from typing import List, Tuple, Dict, Any, Union

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Emergent Representation Mock Backend",
    description="Processes CSV data to generate mocked array outputs based on user-provided logic and documentation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(csv_mock.router)
app.include_router(model_mock.router)
app.include_router(categories_mock.router)

class ProcessedOutput(BaseModel):
    next_point: List[float]
    curve: List[float]
    performance_value: float

class InputVectorPayload(BaseModel):
    vector: List[float]

class ProcessingResult(BaseModel):
    input_row_index: int
    original_input_vector: List[Union[float, str]]
    processed_data: ProcessedOutput
    info: str

class TrainingPayload(BaseModel):
    vector: List[float]
    epochs: int = 10
    iterations: int = 1

class TrainingResult(BaseModel):
    final_result: ProcessingResult
    loss_history: List[float]
    epochs_completed: int

def mock_process_arrays_from_vector(
    input_vector: np.ndarray,
    zeros_length_param: int = -1
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    design = np.array(input_vector, dtype=float)
    info_message = ""

    weights = design * np.random.uniform(0.9, 1.1, size=design.shape)
    weights = np.round(weights, 4)

    if zeros_length_param > 0:
        zeros_length = zeros_length_param
        info_message += f"Using provided zeros_length: {zeros_length}. "
    else:
        zeros_length = max(1, int(np.ceil(len(design) / 2.0)))
        info_message += f"Dynamically set zeros_length: {zeros_length} (half of input length {len(design)}). "

    if len(design) == 3 and np.linalg.norm(design) > 1e-9:
        norm_val = np.linalg.norm(design)
        norm_design = design / norm_val if norm_val > 1e-9 else np.copy(design) 
        
        phi_arg = np.clip(norm_design[2], -1.0, 1.0)
        phi = np.arccos(phi_arg)
        theta = np.arctan2(norm_design[1], norm_design[0])
        
        next_point = np.array([phi, theta])
        info_message += "Processed as 3D Cartesian-like input: next_point is mock 2D Spherical [phi, theta]. "
    elif len(design) == 2:
        phi_in, theta_in = design[0], design[1]
        
        phi = np.clip(phi_in, 0, np.pi) 
        theta = theta_in 

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        next_point = np.array([x, y, z])
        info_message += "Processed as 2D Spherical-like input: next_point is mock 3D Cartesian [x,y,z] (unit norm). "
    elif len(design) > 0 :
        next_point = design * np.random.uniform(0.7, 1.3) + np.random.normal(0, 0.1, size=design.shape)
        info_message += f"Processed as generic {len(design)}D vector: next_point is scaled/noised input. "
    else:
        next_point = np.array([])
        info_message += "Input vector is empty. "

    next_point = np.round(next_point, 4)

    appended_zeros = np.zeros(zeros_length, dtype=float) 
    if len(weights) > 0:
        curve = np.concatenate((weights, appended_zeros))
    else: 
        curve = appended_zeros
    curve = np.round(curve, 4)

    if len(next_point) > 0:
        performance_value = float(np.sum(next_point))
        info_message += f"Performance is sum of next_point elements."
    else:
        performance_value = 0.0
        info_message += f"Performance is 0.0 due to empty next_point."
    
    performance_value = round(performance_value, 4)

    return next_point, curve, performance_value, info_message.strip()

@app.post("/process-csv/", response_model=List[ProcessingResult])
async def process_csv_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    contents = await file.read()
    
    try:
        decoded_contents = contents.decode('utf-8')
    except UnicodeDecodeError:
        try:
            decoded_contents = contents.decode('latin-1') 
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Could not decode CSV file. Ensure it's UTF-8 or Latin-1 encoded.")

    csv_reader = csv.reader(io.StringIO(decoded_contents))
    results: List[ProcessingResult] = []
    
    header_skipped = False
    processed_rows_count = 0

    for i, row_str_list in enumerate(csv_reader):
        current_row_index_for_output = i

        if not row_str_list:
            continue
        
        if not header_skipped and i == 0 and any(char.isalpha() for val in row_str_list for char in val):
            header_skipped = True
            continue 

        input_vector_list_float: List[float] = []
        valid_row = True
        problematic_original_row: List[Union[float, str]] = []

        try:
            for val_str in row_str_list:
                stripped_val = val_str.strip()
                if stripped_val:
                    input_vector_list_float.append(float(stripped_val))
                    problematic_original_row.append(float(stripped_val))
            
            if not input_vector_list_float and any(s.strip() for s in row_str_list): 
                raise ValueError("Row resulted in empty numeric vector after stripping.")

            input_vector_np = np.array(input_vector_list_float)
        except ValueError:
            valid_row = False
            problematic_original_row = [val.strip() for val in row_str_list]

        if not valid_row or not input_vector_list_float :
            results.append(
                ProcessingResult(
                    input_row_index=current_row_index_for_output,
                    original_input_vector=problematic_original_row if problematic_original_row else ["EMPTY_ROW_CONTENT"],
                    processed_data=ProcessedOutput(next_point=[], curve=[], performance_value=0.0),
                    info=f"Skipped row {current_row_index_for_output}: Contains non-numeric data or is effectively empty."
                )
            )
            continue
        
        processed_rows_count +=1
        next_p, crv, perf_val, info = mock_process_arrays_from_vector(input_vector_np)

        results.append(
            ProcessingResult(
                input_row_index=current_row_index_for_output,
                original_input_vector=input_vector_list_float,
                processed_data=ProcessedOutput(
                    next_point=next_p.tolist(),
                    curve=crv.tolist(),
                    performance_value=perf_val
                ),
                info=info
            )
        )

    if processed_rows_count == 0:
        if header_skipped and not results:
            raise HTTPException(status_code=400, detail="CSV file seems to contain only a header or is empty after header.")
        elif not results and not header_skipped :
            raise HTTPException(status_code=400, detail="CSV file is empty or contains no processable numeric data.")
        elif not results and processed_rows_count == 0 :
            raise HTTPException(status_code=400, detail="No processable numeric data found in CSV rows.")

    return results

@app.post("/process-vector/", response_model=ProcessingResult)
async def process_single_vector(payload: InputVectorPayload):
    if not payload.vector:
        raise HTTPException(status_code=400, detail="Input vector cannot be empty.")

    try:
        input_vector_np = np.array(payload.vector, dtype=float)
        
        next_p, crv, perf_val, info = mock_process_arrays_from_vector(input_vector_np)

        result = ProcessingResult(
            input_row_index=0,
            original_input_vector=payload.vector,
            processed_data=ProcessedOutput(
                next_point=next_p.tolist(),
                curve=crv.tolist(),
                performance_value=perf_val
            ),
            info=info
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
    
@app.post("/run-training/", response_model=TrainingResult)
async def run_training_simulation(payload: TrainingPayload):
    if not payload.vector:
        raise HTTPException(status_code=400, detail="Input vector cannot be empty.")
    if not 1 <= payload.epochs <= 1000:
        raise HTTPException(status_code=400, detail="Epochs must be between 1 and 1000.")

    loss_history = []
    current_vector_np = np.array(payload.vector, dtype=float)
    
    initial_loss = np.random.uniform(0.8, 1.5)
    decay_rate = np.random.uniform(0.75, 0.95)

    for epoch in range(payload.epochs):
        noise = np.random.normal(0, 0.05, size=current_vector_np.shape)
        current_vector_np = current_vector_np * np.random.uniform(0.98, 1.02) + noise
        
        loss = initial_loss * (decay_rate ** epoch) + np.random.uniform(-0.05, 0.05)
        loss_history.append(round(max(0.01, loss), 4))

    next_p, crv, perf_val, info = mock_process_arrays_from_vector(current_vector_np)
    
    final_processing_result = ProcessingResult(
        input_row_index=0,
        original_input_vector=payload.vector,
        processed_data=ProcessedOutput(
            next_point=next_p.tolist(),
            curve=crv.tolist(),
            performance_value=perf_val
        ),
        info=f"Result after {payload.epochs} simulated epochs. {info}"
    )

    return TrainingResult(
        final_result=final_processing_result,
        loss_history=loss_history,
        epochs_completed=payload.epochs
    )