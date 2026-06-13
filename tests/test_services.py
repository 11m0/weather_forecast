import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.services.dataset_service import DatasetService
from backend.services.forecast_service import ForecastService


class DatasetServiceTestCase(unittest.TestCase):
    """Тесты локального хранения погодных датасетов."""

    def test_merge_weather_data_removes_duplicate_dates(self) -> None:
        """Проверить объединение записей без дублирования дат."""
        with tempfile.TemporaryDirectory() as directory:
            service = DatasetService()
            service.DATA_DIR = Path(directory)

            service.merge_weather_data(
                records=[
                    {
                        "date": "2026-06-10",
                        "city": "test",
                        "temp_mean": 10.0,
                    }
                ],
                city="test",
            )
            _, added_rows, total_rows = service.merge_weather_data(
                records=[
                    {
                        "date": "2026-06-10",
                        "city": "test",
                        "temp_mean": 11.0,
                    },
                    {
                        "date": "2026-06-11",
                        "city": "test",
                        "temp_mean": 12.0,
                    },
                ],
                city="test",
            )

            dataframe = service.load_weather_data(city="test")

            self.assertEqual(added_rows, 1)
            self.assertEqual(total_rows, 2)
            self.assertEqual(len(dataframe), 2)
            self.assertEqual(dataframe.iloc[0]["temp_mean"], 11.0)


class ForecastServiceTestCase(unittest.TestCase):
    """Тесты сохранения и проверки прогнозов."""

    def test_forecasts_are_appended(self) -> None:
        """Проверить накопление нескольких запусков прогноза."""
        with tempfile.TemporaryDirectory() as directory:
            service = ForecastService()
            service.FORECAST_DIR = Path(directory)
            forecast = [
                {
                    "date": "2026-06-10",
                    "predicted_temp_mean": 10.0,
                }
            ]

            path = service.save_forecast("test", "regression", forecast)
            service.save_forecast("test", "regression", forecast)

            self.assertEqual(len(pd.read_csv(path)), 2)

    def test_compare_with_actual_calculates_metrics(self) -> None:
        """Проверить расчёт MAE и RMSE по совпавшим датам."""
        with tempfile.TemporaryDirectory() as directory:
            service = ForecastService()
            service.FORECAST_DIR = Path(directory)
            service.save_forecast(
                "test",
                "regression",
                [
                    {
                        "date": "2026-06-10",
                        "predicted_temp_mean": 10.0,
                    },
                    {
                        "date": "2026-06-11",
                        "predicted_temp_mean": 14.0,
                    },
                ],
            )
            actual = pd.DataFrame(
                [
                    {"date": "2026-06-10", "temp_mean": 12.0},
                    {"date": "2026-06-11", "temp_mean": 13.0},
                ]
            )

            result = service.compare_with_actual(
                city="test",
                model_name="regression",
                actual_data=actual,
            )

            self.assertEqual(result["metrics"]["mae"], 1.5)
            self.assertEqual(result["metrics"]["rmse"], 1.581)
            self.assertEqual(result["metrics"]["compared_rows"], 2)


if __name__ == "__main__":
    unittest.main()
