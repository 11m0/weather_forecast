from enum import Enum
from typing import Annotated

from fastapi import Query


class CityName(str, Enum):
    """Поддерживаемые идентификаторы городов."""

    MOSCOW = "moscow"
    SAINT_PETERSBURG = "saint_petersburg"
    NOVOSIBIRSK = "novosibirsk"


class ModelName(str, Enum):
    """Поддерживаемые типы моделей прогнозирования."""

    REGRESSION = "regression"
    ARIMA = "arima"
    LSTM = "lstm"


ForecastHorizon = Annotated[
    int,
    Query(ge=1, le=7, description="Горизонт прогноза от 1 до 7 дней."),
]
