from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


class ForecastService:
    """
    Сервис для сохранения прогнозов моделей.
    """

    FORECAST_DIR = Path("data/forecasts")

    def save_forecast(
        self,
        city: str,
        model_name: str,
        forecasts: list[dict],
    ) -> Path:
        """
        Сохранить прогноз модели в CSV.

        Args:
            city: Идентификатор города.
            model_name: Название модели.
            forecasts: Список прогнозов.

        Returns:
            Путь к сохранённому файлу.
        """
        self.FORECAST_DIR.mkdir(parents=True, exist_ok=True)

        file_path = self.FORECAST_DIR / f"{city}_{model_name}_forecast.csv"

        dataframe = pd.DataFrame(forecasts)
        dataframe["city"] = city
        dataframe["model"] = model_name
        dataframe["created_at"] = datetime.now(timezone.utc).isoformat()

        if file_path.exists():
            saved_forecasts = pd.read_csv(file_path)
            dataframe = pd.concat(
                [saved_forecasts, dataframe],
                ignore_index=True,
            )

        dataframe.to_csv(file_path, index=False)

        return file_path

    def compare_with_actual(
        self,
        city: str,
        model_name: str,
        actual_data: pd.DataFrame,
    ) -> dict:
        """
        Сопоставить сохранённые прогнозы с фактической погодой.

        Args:
            city: Идентификатор города.
            model_name: Название модели.
            actual_data: Датасет с фактическими погодными данными.

        Returns:
            Сравнённые точки и рассчитанные метрики MAE и RMSE.

        Raises:
            FileNotFoundError: Если сохранённые прогнозы отсутствуют.
        """
        file_path = self.FORECAST_DIR / f"{city}_{model_name}_forecast.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Forecasts for model '{model_name}' not found."
            )

        forecasts = pd.read_csv(file_path)
        actual = actual_data[["date", "temp_mean"]].copy()
        if "created_at" not in forecasts.columns:
            forecasts["created_at"] = ""
        forecasts["date"] = pd.to_datetime(forecasts["date"])
        actual["date"] = pd.to_datetime(actual["date"])

        comparison = forecasts.merge(actual, on="date", how="inner")
        comparison = comparison.rename(
            columns={
                "predicted_temp_mean": "prediction",
                "temp_mean": "actual",
            }
        )

        if comparison.empty:
            return {
                "points": [],
                "metrics": None,
            }

        comparison["absolute_error"] = (
            comparison["actual"] - comparison["prediction"]
        ).abs()
        errors = comparison["actual"] - comparison["prediction"]
        metrics = {
            "mae": round(float(comparison["absolute_error"].mean()), 3),
            "rmse": round(float(np.sqrt((errors ** 2).mean())), 3),
            "compared_rows": len(comparison),
        }
        comparison["date"] = comparison["date"].dt.strftime("%Y-%m-%d")
        points = comparison[
            [
                "date",
                "prediction",
                "actual",
                "absolute_error",
                "created_at",
            ]
        ].sort_values(["date", "created_at"])

        return {
            "points": points.to_dict(orient="records"),
            "metrics": metrics,
        }
