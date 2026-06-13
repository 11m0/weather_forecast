import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend.services.automation_service import AutomationService


class AutomationServiceTestCase(unittest.TestCase):
    """Тесты фонового обновления погодных данных."""

    def setUp(self) -> None:
        """Создать изолированные каталоги автоматизации."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.service = AutomationService(interval_seconds=60)
        self.service.dataset_service.DATA_DIR = root / "processed"
        self.service.forecast_service.FORECAST_DIR = root / "forecasts"
        self.service.metrics_service.METRICS_DIR = root / "metrics"
        self.service.STATUS_DIR = root / "automation"
        self.service.STATUS_FILE = self.service.STATUS_DIR / "status.json"

    def tearDown(self) -> None:
        """Удалить временные каталоги после теста."""
        self.temporary_directory.cleanup()

    def test_get_existing_cities_returns_saved_datasets(self) -> None:
        """Проверить выбор только городов с локальными датасетами."""
        self.service.dataset_service.DATA_DIR.mkdir(parents=True)
        pd.DataFrame(
            [{"date": "2026-06-10", "temp_mean": 10.0}]
        ).to_csv(
            self.service.dataset_service.DATA_DIR / "moscow_weather.csv",
            index=False,
        )

        self.assertEqual(self.service.get_existing_cities(), ["moscow"])

    def test_run_once_saves_status(self) -> None:
        """Проверить сохранение результата одного цикла."""
        with patch.object(
            self.service,
            "get_existing_cities",
            return_value=["moscow"],
        ), patch.object(
            self.service,
            "update_city_data",
            return_value={"added_rows": 0, "total_rows": 100},
        ), patch.object(
            self.service,
            "update_comparison_metrics",
            return_value={"regression": {"mae": 1.0, "rmse": 2.0}},
        ):
            result = self.service.run_once()

        saved_status = self.service.load_status()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(saved_status["status"], "ok")
        self.assertIn("moscow", saved_status["cities"])

    def test_comparison_metrics_are_saved(self) -> None:
        """Проверить сохранение постфактум метрик модели."""
        self.service.dataset_service.DATA_DIR.mkdir(parents=True)
        pd.DataFrame(
            [{"date": "2026-06-10", "temp_mean": 12.0}]
        ).to_csv(
            self.service.dataset_service.DATA_DIR / "moscow_weather.csv",
            index=False,
        )
        self.service.forecast_service.save_forecast(
            city="moscow",
            model_name="regression",
            forecasts=[
                {
                    "date": "2026-06-10",
                    "predicted_temp_mean": 10.0,
                }
            ],
        )

        result = self.service.update_comparison_metrics("moscow")
        metrics_path = (
            self.service.metrics_service.METRICS_DIR
            / "moscow_regression_comparison_metrics.csv"
        )

        self.assertEqual(result["regression"]["mae"], 2.0)
        self.assertTrue(metrics_path.exists())


if __name__ == "__main__":
    unittest.main()
