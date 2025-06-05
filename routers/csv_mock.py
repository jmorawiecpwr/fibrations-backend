from fastapi import APIRouter
from models.schemas import CSVResponse

router = APIRouter()

@router.get("/csv", response_model=CSVResponse)
def get_mock_csv():
    return {
        "headers": ["Column 1", "Column 2", "Column 3"],
        "rows": [
            ["a", 1, "w"],
            ["b", 2, "x"],
            ["c", 3, "y"],
            ["d", 4, "z"],
            ["e", 5, "Lorem ipsum"]
        ]
    }