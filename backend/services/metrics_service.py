from pathlib import Path

import pandas as pd


class MetricsService:
    """
    Сервис для сохранения и загрузки метрик моделей.
    """

    METRICS_DIR = Path("data/metrics")

    def save_metrics(
        self,
        city: str,
        model_name: str,
        metrics: dict,
    ) -> Path:
        """
        Сохранить метрики модели в CSV.

        Args:
            city: Идентификатор города.
            model_name: Название модели.
            metrics: Метрики модели.

        Returns:
            Путь к сохранённому файлу.
        """
        self.METRICS_DIR.mkdir(parents=True, exist_ok=True)

        file_path = self.METRICS_DIR / f"{city}_{model_name}_metrics.csv"

        dataframe = pd.DataFrame(
            [
                {
                    "city": city,
                    "model": model_name,
                    **metrics,
                }
            ]
        )

        dataframe.to_csv(file_path, index=False)

        return file_path

    def load_metrics(
            self,
            city: str,
            model_name: str,
    ) -> dict:
        """
        Загрузить метрики модели.

        Args:
            city: Идентификатор города.
            model_name: Название модели.

        Returns:
            Словарь с метриками.
        """
        file_path = (
                self.METRICS_DIR /
                f"{city}_{model_name}_metrics.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Metrics for model '{model_name}' not found."
            )

        dataframe = pd.read_csv(file_path)

        return dataframe.iloc[0].to_dict()

    def save_comparison_metrics(
        self,
        city: str,
        model_name: str,
        metrics: dict,
    ) -> Path:
        """
        Сохранить постфактум метрики прогноза.

        Args:
            city: Идентификатор города.
            model_name: Название модели.
            metrics: Метрики сравнения прогноза с фактом.

        Returns:
            Путь к сохранённому CSV-файлу.
        """
        self.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = (
            self.METRICS_DIR
            / f"{city}_{model_name}_comparison_metrics.csv"
        )
        dataframe = pd.DataFrame(
            [{"city": city, "model": model_name, **metrics}]
        )
        dataframe.to_csv(file_path, index=False)
        return file_path
