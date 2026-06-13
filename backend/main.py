import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.automation import router as automation_router
from backend.api.cities import router as cities_router
from backend.api.models import router as models_router, train_and_save_model
from backend.api.weather import router as weather_router
from backend.services.automation_service import AutomationService


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Запустить и остановить фоновую задачу автоматизации.

    Args:
        application: Экземпляр FastAPI-приложения.
    """
    automation_task = None
    automation_enabled = (
        os.getenv("AUTOMATION_ENABLED", "false").lower() == "true"
    )

    if automation_enabled:
        automation_service = AutomationService.from_environment()
        application.state.automation_service = automation_service
        automation_task = asyncio.create_task(
            automation_service.run_forever(train_and_save_model)
        )

    yield

    if automation_task is not None:
        automation_task.cancel()
        try:
            await automation_task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="Weather Forecast Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cities_router)
app.include_router(models_router)
app.include_router(weather_router)
app.include_router(automation_router)


@app.get("/health")
def health_check() -> dict:
    """
    Проверить состояние API.
    """
    return {"status": "ok"}
