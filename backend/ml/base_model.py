from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseWeatherModel(ABC):
    """
    Базовый интерфейс для всех погодных моделей.
    """

    MODEL_DIR = Path("models")

    @abstractmethod
    def train(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        """
        Обучить модель.

        Args:
            dataframe: Датасет для обучения.

        Returns:
            Метрики качества модели.
        """

    @abstractmethod
    def predict(self, dataframe: pd.DataFrame) -> list[float]:
        """
        Выполнить предсказание.

        Args:
            dataframe: Датасет с признаками.

        Returns:
            Список предсказанных значений.
        """

    @abstractmethod
    def forecast(
        self,
        dataframe: pd.DataFrame,
        feature_builder: Any,
        horizon: int,
    ) -> list[dict]:
        """
        Построить прогноз на несколько дней вперёд.

        Args:
            dataframe: Исторические данные.
            feature_builder: Объект для построения признаков.
            horizon: Горизонт прогноза.

        Returns:
            Список прогнозов.
        """

    @abstractmethod
    def save(self, city: str) -> Path:
        """
        Сохранить модель.

        Args:
            city: Идентификатор города.

        Returns:
            Путь к сохранённой модели.
        """

    @abstractmethod
    def load(self, city: str) -> None:
        """
        Загрузить модель.

        Args:
            city: Идентификатор города.
        """
