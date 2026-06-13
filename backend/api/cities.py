from fastapi import APIRouter

from backend.config import CITIES

router = APIRouter(prefix="/cities", tags=["cities"])


@router.get("")
def get_cities() -> dict:
    """Получить список поддерживаемых городов."""
    return CITIES
