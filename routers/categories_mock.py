from fastapi import APIRouter

router = APIRouter()

@router.get("/categories")
def get_categories():
    return ["Alloy-Alpha", "Titanium-X", "Ceramic-Z", "Carbon-Mock"]