import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from backend.config import CITIES
from backend.services.model_service import ModelService


class AutomationService:
    """Сервис фонового обновления данных и проверки прогнозов."""

    STATUS_DIR = Path("data/automation")
    STATUS_FILE = STATUS_DIR / "status.json"

    def __init__(
        self,
        interval_seconds: int = 86400,
        retrain_enabled: bool = False,
    ) -> None:
        """
        Инициализировать сервис автоматизации.

        Args:
            interval_seconds: Интервал между запусками в секундах.
            retrain_enabled: Нужно ли переобучать сохранённые модели.
        """
        self.interval_seconds = max(interval_seconds, 60)
        self.retrain_enabled = retrain_enabled
        self.model_service = ModelService()
        self.dataset_service = self.model_service.dataset_service
        self.forecast_service = self.model_service.forecast_service
        self.metrics_service = self.model_service.metrics_service

    @classmethod
    def from_environment(cls) -> "AutomationService":
        """
        Создать сервис из переменных окружения.

        Returns:
            Настроенный сервис автоматизации.
        """
        interval_seconds = int(
            os.getenv("AUTOMATION_INTERVAL_SECONDS", "86400")
        )
        retrain_enabled = (
            os.getenv("AUTOMATION_RETRAIN_ENABLED", "false").lower()
            == "true"
        )
        return cls(
            interval_seconds=interval_seconds,
            retrain_enabled=retrain_enabled,
        )

    def get_existing_cities(self) -> list[str]:
        """
        Получить города с уже сохранёнными датасетами.

        Returns:
            Список идентификаторов городов.
        """
        return [
            city
            for city in CITIES
            if (self.dataset_service.DATA_DIR / f"{city}_weather.csv").exists()
        ]

    def update_city_data(self, city: str) -> dict:
        """
        Догрузить отсутствующие погодные данные города до вчерашнего дня.

        Args:
            city: Идентификатор города.

        Returns:
            Количество добавленных и общее количество записей.
        """
        result = self.model_service.update_data(city)
        return {
            "added_rows": result["added_rows"],
            "total_rows": result["total_rows"],
        }

    def update_comparison_metrics(self, city: str) -> dict:
        """
        Пересчитать постфактум метрики всех моделей города.

        Args:
            city: Идентификатор города.

        Returns:
            Метрики моделей, для которых уже доступен факт.
        """
        dataframe = self.dataset_service.load_weather_data(city=city)
        result = {}

        for model_name in ("regression", "arima", "lstm"):
            try:
                comparison = self.forecast_service.compare_with_actual(
                    city=city,
                    model_name=model_name,
                    actual_data=dataframe,
                )
            except FileNotFoundError:
                continue

            if comparison["metrics"] is not None:
                self.metrics_service.save_comparison_metrics(
                    city=city,
                    model_name=model_name,
                    metrics=comparison["metrics"],
                )
                result[model_name] = comparison["metrics"]

        return result

    def run_once(
        self,
        train_callback: Callable[[str, str], dict] | None = None,
    ) -> dict:
        """
        Выполнить один цикл обновления и проверки прогнозов.

        Args:
            train_callback: Функция переобучения модели.

        Returns:
            Статус выполненного цикла по городам.
        """
        started_at = datetime.now(timezone.utc)
        result = {
            "status": "ok",
            "started_at": started_at.isoformat(),
            "finished_at": None,
            "interval_seconds": self.interval_seconds,
            "retrain_enabled": self.retrain_enabled,
            "cities": {},
        }

        for city in self.get_existing_cities():
            city_result = {}
            try:
                city_result["data"] = self.update_city_data(city)
                city_result["comparison_metrics"] = (
                    self.update_comparison_metrics(city)
                )

                if self.retrain_enabled and train_callback is not None:
                    city_result["training"] = {}
                    for model_name in ("regression", "arima", "lstm"):
                        city_result["training"][model_name] = train_callback(
                            city,
                            model_name,
                        )["metrics"]
            except (FileNotFoundError, KeyError, ValueError,
                    requests.RequestException) as error:
                result["status"] = "partial_error"
                city_result["error"] = str(error)

            result["cities"][city] = city_result

        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.save_status(result)
        return result

    async def run_forever(
        self,
        train_callback: Callable[[str, str], dict] | None = None,
    ) -> None:
        """
        Периодически выполнять цикл автоматизации до отмены задачи.

        Args:
            train_callback: Функция переобучения модели.
        """
        while True:
            await asyncio.to_thread(self.run_once, train_callback)
            await asyncio.sleep(self.interval_seconds)

    def save_status(self, status: dict) -> Path:
        """
        Сохранить статус последнего автоматического запуска.

        Args:
            status: Данные о выполненном цикле.

        Returns:
            Путь к JSON-файлу статуса.
        """
        self.STATUS_DIR.mkdir(parents=True, exist_ok=True)
        self.STATUS_FILE.write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.STATUS_FILE

    def load_status(self) -> dict:
        """
        Загрузить статус последнего автоматического запуска.

        Returns:
            Сохранённый статус или информацию об отсутствии запусков.
        """
        if not self.STATUS_FILE.exists():
            return {
                "status": "not_run",
                "finished_at": None,
                "interval_seconds": self.interval_seconds,
                "retrain_enabled": self.retrain_enabled,
                "cities": {},
            }

        return json.loads(self.STATUS_FILE.read_text(encoding="utf-8"))
