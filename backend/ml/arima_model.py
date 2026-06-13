from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from backend.ml.base_model import BaseWeatherModel


class ArimaModel(BaseWeatherModel):
    """
    ARIMA-модель для прогнозирования температуры.
    """

    MODEL_ORDER = (5, 1, 0)

    def __init__(self) -> None:
        """
        Инициализировать ARIMA-модель.
        """
        self.model = None
        self.fitted_model = None

    def train(self, dataframe: pd.DataFrame) -> dict:
        """
        Обучить ARIMA-модель.

        Args:
            dataframe: Исторический датасет с погодными данными.

        Returns:
            Метрики обучения.
        """
        if len(dataframe) < 2:
            raise ValueError(
                "Для обучения ARIMA требуется минимум 2 записи."
            )

        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = dataframe.sort_values("date")

        series = dataframe.set_index("date")["temp_mean"]
        series = series.asfreq("D")

        train_size = int(len(series) * 0.8)

        train_series = series.iloc[:train_size]
        test_series = series.iloc[train_size:]

        self.model = ARIMA(
            train_series,
            order=self.MODEL_ORDER,
        )

        self.fitted_model = self.model.fit()

        predictions = self.fitted_model.forecast(
            steps=len(test_series),
        )

        mae = mean_absolute_error(
            test_series,
            predictions,
        )

        mse = mean_squared_error(
            test_series,
            predictions,
        )

        rmse = mse ** 0.5

        self.model = ARIMA(
            series,
            order=self.MODEL_ORDER,
        )
        self.fitted_model = self.model.fit()

        return {
            "model": "arima",
            "order": self.MODEL_ORDER,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "train_rows": len(train_series),
            "test_rows": len(test_series),
        }

    def predict(self, dataframe: pd.DataFrame) -> list[float]:
        """
        Выполнить предсказание ARIMA на переданном временном ряду.

        Args:
            dataframe: Датасет с колонками date и temp_mean.

        Returns:
            Список предсказанных значений.
        """
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = dataframe.sort_values("date")

        start = dataframe["date"].min()
        end = dataframe["date"].max()

        predictions = self.fitted_model.predict(
            start=start,
            end=end,
        )

        return [round(float(value), 2) for value in predictions]

    def forecast(
            self,
            dataframe: pd.DataFrame,
            feature_builder,
            horizon: int,
    ) -> list[dict]:
        """
        Построить прогноз ARIMA на несколько дней вперёд.

        Args:
            dataframe: Исторические данные.
            feature_builder: Не используется для ARIMA.
            horizon: Горизонт прогноза.

        Returns:
            Список прогнозов.
        """
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = dataframe.sort_values("date")

        last_date = dataframe["date"].max()
        predicted_values = self.fitted_model.forecast(steps=horizon)

        forecasts = []

        for index, value in enumerate(predicted_values, start=1):
            forecast_date = last_date + pd.Timedelta(days=index)

            forecasts.append(
                {
                    "date": str(forecast_date.date()),
                    "predicted_temp_mean": round(float(value), 2),
                }
            )

        return forecasts

    def save(self, city: str) -> Path:
        """
        Сохранить обученную ARIMA-модель.

        Args:
            city: Идентификатор города.

        Returns:
            Путь к сохранённой модели.
        """
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        file_path = self.MODEL_DIR / f"{city}_arima_model.joblib"
        joblib.dump(self.fitted_model, file_path)

        return file_path

    def load(self, city: str) -> None:
        """
        Загрузить обученную ARIMA-модель.

        Args:
            city: Идентификатор города.
        """
        file_path = self.MODEL_DIR / f"{city}_arima_model.joblib"
        self.fitted_model = joblib.load(file_path)
