from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from backend.ml.base_model import BaseWeatherModel


class RegressionModel(BaseWeatherModel):
    """
    Табличная ML-модель для прогноза средней дневной температуры.
    """

    TARGET_COLUMN = "temp_mean"

    FEATURE_COLUMNS = [
        "month",
        "day_of_year",
        "day_of_week",
        "temp_lag_1",
        "temp_lag_2",
        "temp_lag_3",
        "temp_lag_7",
        "temp_rolling_3",
        "temp_rolling_7",
    ]

    def __init__(self) -> None:
        """
        Инициализировать RandomForestRegressor.
        """
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        )

    def train(self, dataframe: pd.DataFrame) -> dict:
        """
        Обучить модель на датасете с признаками.

        Args:
            dataframe: Датасет с признаками и целевой переменной.

        Returns:
            Метрики качества модели.
        """
        if len(dataframe) < 2:
            raise ValueError(
                "Для обучения регрессии требуется минимум 2 записи."
            )

        train_size = int(len(dataframe) * 0.8)

        train_data = dataframe.iloc[:train_size]
        test_data = dataframe.iloc[train_size:]

        x_train = train_data[self.FEATURE_COLUMNS]
        y_train = train_data[self.TARGET_COLUMN]

        x_test = test_data[self.FEATURE_COLUMNS]
        y_test = test_data[self.TARGET_COLUMN]

        self.model.fit(x_train, y_train)

        predictions = self.model.predict(x_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5

        self.model.fit(
            dataframe[self.FEATURE_COLUMNS],
            dataframe[self.TARGET_COLUMN],
        )

        return {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "train_rows": len(train_data),
            "test_rows": len(test_data),
        }

    def save(self, city: str) -> Path:
        """
        Сохранить обученную модель в файл.

        Args:
            city: Идентификатор города.

        Returns:
            Путь к сохранённой модели.
        """
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        file_path = self.MODEL_DIR / f"{city}_regression_model.joblib"
        joblib.dump(self.model, file_path)

        return file_path

    def load(self, city: str) -> None:
        """
        Загрузить обученную модель из файла.

        Args:
            city: Идентификатор города.
        """
        file_path = self.MODEL_DIR / f"{city}_regression_model.joblib"
        self.model = joblib.load(file_path)

    def predict(self, dataframe: pd.DataFrame) -> list[float]:
        """
        Сделать прогноз средней температуры.

        Args:
            dataframe: Датасет с признаками.

        Returns:
            Список предсказанных значений.
        """
        features = dataframe[self.FEATURE_COLUMNS]
        predictions = self.model.predict(features)

        return [round(float(value), 2) for value in predictions]

    def forecast(
            self,
            dataframe: pd.DataFrame,
            feature_builder,
            horizon: int,
    ) -> list[dict]:
        """
        Построить прогноз на несколько дней вперёд.

        Args:
            dataframe: Исторические данные.
            feature_builder: Построитель признаков.
            horizon: Горизонт прогноза.

        Returns:
            Список прогнозов.
        """
        history = dataframe.copy()
        forecasts = []

        for _ in range(horizon):
            next_features = feature_builder.build_next_day_features(history)

            predicted_temp = self.predict(next_features)[0]
            forecast_date = next_features["date"].iloc[0]

            forecasts.append(
                {
                    "date": str(forecast_date.date()),
                    "predicted_temp_mean": predicted_temp,
                }
            )

            new_row = {
                "date": forecast_date,
                "temp_mean": predicted_temp,
            }

            history = pd.concat(
                [history, pd.DataFrame([new_row])],
                ignore_index=True,
            )

        return forecasts
