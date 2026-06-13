from fastapi import APIRouter

from backend.services.automation_service import AutomationService


router = APIRouter(
    prefix="/automation",
    tags=["automation"],
)

automation_service = AutomationService.from_environment()


@router.get("/status")
def get_automation_status() -> dict:
    """Получить статус последнего автоматического обновления."""
    return automation_service.load_status()
