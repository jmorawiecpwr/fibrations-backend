from pydantic import BaseModel, Field
from typing import Optional, List, Any

class CSVResponse(BaseModel):
    headers: List[str] = Field(..., description="List of column headers")
    rows: List[List[Any]] = Field(..., description="List of rows, each row is a list of values")