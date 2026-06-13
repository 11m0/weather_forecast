from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Dense
from keras.models import Sequential, load_model

from backend.ml.base_model import BaseWeatherModel


class LstmModel(BaseWeatherModel):
    """
    LSTM-модель для прогнозирования температуры.
    """
    WINDOW_SIZE = 14

    def __init__(self) -> None:
        """
        Инициализировать LSTM-модель.
        """
        self.model = None

    def train(self, dataframe: pd.DataFrame) -> dict:
        """
        Обучить LSTM-модель.

        Args:
            dataframe: Исторический датасет с погодными данными.

        Returns:
            Метрики обучения.
        """
        if len(dataframe) <= self.WINDOW_SIZE:
            raise ValueError(
                "Для обучения LSTM требуется больше "
                f"{self.WINDOW_SIZE} записей."
            )

        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = dataframe.sort_values("date")

        series = dataframe["temp_mean"].to_numpy(dtype=float)

        x_data, y_data = self.create_sequences(
            series=series,
            window_size=self.WINDOW_SIZE,
        )

        x_data = x_data.reshape(
            (x_data.shape[0], x_data.shape[1], 1)
        )

        train_size = int(len(x_data) * 0.8)

        x_train = x_data[:train_size]
        y_train = y_data[:train_size]

        x_test = x_data[train_size:]
        y_test = y_data[train_size:]

        self.model = Sequential(
            [
                Input(shape=(self.WINDOW_SIZE, 1)),
                LSTM(32),
                Dense(1),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="mse",
        )

        self.model.fit(
            x_train,
            y_train,
            epochs=20,
            batch_size=16,
            verbose=0,
        )

        predictions = self.model.predict(x_test, verbose=0).flatten()

        mae = float(np.mean(np.abs(y_test - predictions)))
        rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))

        self.model.fit(
            x_data,
            y_data,
            epochs=20,
            batch_size=16,
            verbose=0,
        )

        return {
            "model": "lstm",
            "window_size": self.WINDOW_SIZE,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
        }

    def predict(self, dataframe: pd.DataFrame) -> list[float]:
        """
        Выполнить предсказание для набора данных.

        Args:
            dataframe: Исторический датасет.

        Returns:
            Список предсказаний.
        """
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = dataframe.sort_values("date")

        series = dataframe["temp_mean"].to_numpy(dtype=float)

        x_data, _ = self.create_sequences(
            series=series,
            window_size=self.WINDOW_SIZE,
        )

        x_data = x_data.reshape(
            (x_data.shape[0], x_data.shape[1], 1)
        )

        predictions = self.model.predict(
            x_data,
            verbose=0,
        ).flatten()

        return [
            round(float(value), 2)
            for value in predictions
        ]

    def forecast(
            self,
            dataframe: pd.DataFrame,
            feature_builder,
            horizon: int,
    ) -> list[dict]:
        """
        Построить прогноз LSTM на несколько дней вперёд.

        Args:
            dataframe: Исторические данные.
            feature_builder: Не используется для LSTM.
            horizon: Горизонт прогноза.

        Returns:
            Список прогнозов.
        """
        if len(dataframe) < self.WINDOW_SIZE:
            raise ValueError(
                "Для прогноза LSTM требуется минимум "
                f"{self.WINDOW_SIZE} записей."
            )

        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = dataframe.sort_values("date")

        history = dataframe["temp_mean"].to_list()
        last_date = dataframe["date"].max()

        forecasts = []

        for step in range(1, horizon + 1):
            window = np.array(history[-self.WINDOW_SIZE:], dtype=float)
            x_input = window.reshape((1, self.WINDOW_SIZE, 1))

            prediction = self.model.predict(
                x_input,
                verbose=0,
            )[0][0]

            prediction = round(float(prediction), 2)
            forecast_date = last_date + pd.Timedelta(days=step)

            forecasts.append(
                {
                    "date": str(forecast_date.date()),
                    "predicted_temp_mean": prediction,
                }
            )

            history.append(prediction)

        return forecasts

    def save(self, city: str) -> Path:
        """
        Сохранить обученную LSTM-модель.

        Args:
            city: Идентификатор города.

        Returns:
            Путь к сохранённой модели.
        """
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        file_path = self.MODEL_DIR / f"{city}_lstm_model.keras"
        self.model.save(file_path)

        return file_path

    def load(self, city: str) -> None:
        """
        Загрузить обученную LSTM-модель.

        Args:
            city: Идентификатор города.
        """
        file_path = self.MODEL_DIR / f"{city}_lstm_model.keras"
        self.model = load_model(file_path)

    @staticmethod
    def create_sequences(
            series: np.ndarray,
            window_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Создать обучающие последовательности для LSTM.

        Args:
            series: Временной ряд.
            window_size: Размер окна.

        Returns:
            X и y для обучения.
        """
        x_data = []
        y_data = []

        for index in range(len(series) - window_size):
            x_data.append(
                series[index:index + window_size]
            )

            y_data.append(
                series[index + window_size]
            )

        return (
            np.array(x_data),
            np.array(y_data),
        )
