import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api import models as models_api
from backend.main import app


class ApiValidationTestCase(unittest.TestCase):
    """Тесты валидации параметров публичного API."""

    @classmethod
    def setUpClass(cls) -> None:
        """Создать тестовый HTTP-клиент FastAPI."""
        cls.client = TestClient(app)

    def setUp(self) -> None:
        """Изолировать локальный каталог датасетов для каждого теста."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_data_dir = models_api.dataset_service.DATA_DIR
        models_api.dataset_service.DATA_DIR = Path(
            self.temporary_directory.name
        )

    def tearDown(self) -> None:
        """Восстановить каталог датасетов после теста."""
        models_api.dataset_service.DATA_DIR = self.original_data_dir
        self.temporary_directory.cleanup()

    def test_unknown_city_returns_validation_error(self) -> None:
        """Проверить отклонение неизвестного города."""
        response = self.client.get(
            "/weather/dataset",
            params={"city": "london"},
        )

        self.assertEqual(response.status_code, 422)

    def test_unknown_model_returns_validation_error(self) -> None:
        """Проверить отклонение неизвестной модели."""
        response = self.client.get(
            "/models/forecast",
            params={
                "city": "moscow",
                "model_name": "unknown",
                "horizon": 3,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_invalid_horizon_returns_validation_error(self) -> None:
        """Проверить ограничение горизонта диапазоном от 1 до 7."""
        response = self.client.get(
            "/models/forecast",
            params={
                "city": "moscow",
                "model_name": "regression",
                "horizon": 8,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_reversed_date_range_returns_validation_error(self) -> None:
        """Проверить отклонение диапазона с обратным порядком дат."""
        response = self.client.get(
            "/weather/history",
            params={
                "city": "moscow",
                "start_date": "2026-06-10",
                "end_date": "2026-06-01",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("Начальная дата", response.json()["detail"])

    def test_train_returns_not_found_without_dataset(self) -> None:
        """Проверить понятную ошибку обучения без датасета."""
        response = self.client.post(
            "/models/train",
            params={
                "city": "moscow",
                "model_name": "regression",
            },
        )

        self.assertEqual(response.status_code, 404)

    def test_train_uses_common_endpoint(self) -> None:
        """Проверить успешный ответ общего endpoint обучения."""
        training_result = {
            "metrics": {"mae": 1.0, "rmse": 2.0},
            "model_path": "models/test.joblib",
            "metrics_path": "data/metrics/test.csv",
        }

        with patch(
            "backend.api.models.train_and_save_model",
            return_value=training_result,
        ):
            response = self.client.post(
                "/models/train",
                params={
                    "city": "moscow",
                    "model_name": "regression",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "regression")
        self.assertEqual(response.json()["metrics"]["mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
