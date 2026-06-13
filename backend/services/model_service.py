from datetime import date, timedelta

import pandas as pd

from backend.config import CITIES
from backend.ml.feature_builder import FeatureBuilder
from backend.ml.lstm_model import LstmModel
from backend.ml.model_factory import ModelFactory
from backend.ml.regression_model import RegressionModel
from backend.services.dataset_service import DatasetService
from backend.services.forecast_service import ForecastService
from backend.services.metrics_service import MetricsService
from backend.services.weather_service import WeatherService


class ModelService:
    """Сервис обучения моделей, прогнозирования и обновления данных."""

    MODEL_NAMES = ("regression", "arima", "lstm")

    def __init__(self) -> None:
        """Инициализировать зависимости сервиса моделей."""
        self.dataset_service = DatasetService()
        self.feature_builder = FeatureBuilder()
        self.forecast_service = ForecastService()
        self.metrics_service = MetricsService()
        self.weather_service = WeatherService()

    def validate_training_data(
        self,
        dataframe: pd.DataFrame,
        model_name: str,
    ) -> None:
        """
        Проверить датасет перед обучением модели.

        Args:
            dataframe: Подготовленные данные модели.
            model_name: Название модели.

        Raises:
            ValueError: Если обязательных данных недостаточно.
        """
        required_columns = {"date", "temp_mean"}
        if model_name == "regression":
            required_columns.update(RegressionModel.FEATURE_COLUMNS)

        self._validate_columns(dataframe, required_columns)
        minimum_rows = {
            "regression": 10,
            "arima": 20,
            "lstm": LstmModel.WINDOW_SIZE + 5,
        }[model_name]
        self._validate_row_count(
            dataframe,
            minimum_rows,
            f"обучения модели '{model_name}'",
        )

    def validate_forecast_data(
        self,
        dataframe: pd.DataFrame,
        model_name: str,
    ) -> None:
        """
        Проверить датасет перед построением прогноза.

        Args:
            dataframe: Исторические погодные данные.
            model_name: Название модели.

        Raises:
            ValueError: Если данных недостаточно для прогноза.
        """
        required_columns = {"date", "temp_mean"}
        self._validate_columns(dataframe, required_columns)
        minimum_rows = {
            "regression": 7,
            "arima": 1,
            "lstm": LstmModel.WINDOW_SIZE,
        }[model_name]
        self._validate_row_count(
            dataframe,
            minimum_rows,
            f"прогноза модели '{model_name}'",
        )

    def train(self, city: str, model_name: str) -> dict:
        """
        Обучить модель и сохранить артефакты и метрики.

        Args:
            city: Идентификатор города.
            model_name: Название модели.

        Returns:
            Метрики и пути к сохранённым файлам.
        """
        dataframe = self.dataset_service.load_weather_data(city=city)
        model = ModelFactory.create(model_name)
        training_data = (
            self.feature_builder.build_features(dataframe)
            if model_name == "regression"
            else dataframe
        )
        self.validate_training_data(training_data, model_name)
        metrics = model.train(training_data)
        model_path = model.save(city=city)
        metrics_path = self.metrics_service.save_metrics(
            city=city,
            model_name=model_name,
            metrics=metrics,
        )

        return {
            "metrics": metrics,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        }

    def forecast(
        self,
        city: str,
        model_name: str,
        horizon: int,
    ) -> dict:
        """
        Построить и сохранить прогноз выбранной модели.

        Args:
            city: Идентификатор города.
            model_name: Название модели.
            horizon: Горизонт прогноза в днях.

        Returns:
            Прогноз и путь к сохранённому файлу.
        """
        dataframe = self.dataset_service.load_weather_data(city=city)
        self.validate_forecast_data(dataframe, model_name)
        model = ModelFactory.create(model_name)
        model.load(city=city)
        forecasts = model.forecast(
            dataframe=dataframe,
            feature_builder=self.feature_builder,
            horizon=horizon,
        )
        forecast_path = self.forecast_service.save_forecast(
            city=city,
            model_name=model_name,
            forecasts=forecasts,
        )

        return {
            "forecast": forecasts,
            "forecast_path": str(forecast_path),
        }

    def update_data(self, city: str) -> dict:
        """
        Обновить локальный датасет города до вчерашнего дня.

        Args:
            city: Идентификатор города.

        Returns:
            Сведения о количестве сохранённых записей.
        """
        end_date = date.today() - timedelta(days=1)

        try:
            dataframe = self.dataset_service.load_weather_data(city=city)
            last_date = date.fromisoformat(str(dataframe["date"].max()))
            start_date = last_date + timedelta(days=1)
        except FileNotFoundError:
            dataframe = None
            start_date = end_date - timedelta(days=3 * 365)

        dataset_path = (
            self.dataset_service.DATA_DIR / f"{city}_weather.csv"
        )
        if start_date > end_date:
            return {
                "added_rows": 0,
                "total_rows": len(dataframe),
                "dataset_path": str(dataset_path),
            }

        city_data = CITIES[city]
        raw_data = self.weather_service.get_historical_weather(
            latitude=city_data["latitude"],
            longitude=city_data["longitude"],
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        records = self.weather_service.normalize_daily_weather(
            raw_data=raw_data,
            city=city,
        )
        dataset_path, added_rows, total_rows = (
            self.dataset_service.merge_weather_data(
                records=records,
                city=city,
            )
        )

        return {
            "added_rows": added_rows,
            "total_rows": total_rows,
            "dataset_path": str(dataset_path),
        }

    def update_and_train(self, city: str, model_name: str) -> dict:
        """
        Обновить данные и переобучить выбранную модель.

        Args:
            city: Идентификатор города.
            model_name: Название модели.

        Returns:
            Результаты обновления и обучения.
        """
        data_result = self.update_data(city)
        training_result = self.train(city, model_name)
        return {
            "data": data_result,
            **training_result,
        }

    def compare_with_actual(self, city: str, model_name: str) -> dict:
        """
        Сопоставить сохранённые прогнозы с фактическими данными.

        Args:
            city: Идентификатор города.
            model_name: Название модели.

        Returns:
            Точки сравнения и постфактум метрики.
        """
        try:
            dataframe = self.dataset_service.load_weather_data(city=city)
            return self.forecast_service.compare_with_actual(
                city=city,
                model_name=model_name,
                actual_data=dataframe,
            )
        except FileNotFoundError:
            return {"points": [], "metrics": None}

    def predict_history(self, city: str) -> list[dict]:
        """
        Получить предсказания regression для сохранённой истории.

        Args:
            city: Идентификатор города.

        Returns:
            Фактические и предсказанные значения по датам.
        """
        dataframe = self.dataset_service.load_weather_data(city=city)
        features = self.feature_builder.build_features(dataframe)
        model = ModelFactory.create("regression")
        model.load(city=city)
        result = features[["date", "temp_mean"]].copy()
        result["prediction"] = model.predict(features)
        return result.to_dict(orient="records")

    def get_metrics(self, city: str) -> dict:
        """
        Получить метрики всех обученных моделей города.

        Args:
            city: Идентификатор города.

        Returns:
            Словарь метрик по названиям моделей.
        """
        result = {}
        for model_name in self.MODEL_NAMES:
            try:
                result[model_name] = self.metrics_service.load_metrics(
                    city=city,
                    model_name=model_name,
                )
            except FileNotFoundError:
                result[model_name] = None
        return result

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        required_columns: set[str],
    ) -> None:
        """
        Проверить наличие и заполненность обязательных колонок.

        Args:
            dataframe: Проверяемый датасет.
            required_columns: Обязательные названия колонок.

        Raises:
            ValueError: Если колонки отсутствуют или содержат пропуски.
        """
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"В датасете отсутствуют колонки: {columns}.")

        if dataframe[list(required_columns)].isnull().any().any():
            raise ValueError(
                "Датасет содержит пропуски в обязательных колонках."
            )

    @staticmethod
    def _validate_row_count(
        dataframe: pd.DataFrame,
        minimum_rows: int,
        operation: str,
    ) -> None:
        """
        Проверить минимальное количество строк датасета.

        Args:
            dataframe: Проверяемый датасет.
            minimum_rows: Минимально допустимое число строк.
            operation: Название выполняемой операции.

        Raises:
            ValueError: Если строк недостаточно.
        """
        if len(dataframe) < minimum_rows:
            raise ValueError(
                f"Для {operation} требуется минимум {minimum_rows} "
                f"записей, получено {len(dataframe)}."
            )
