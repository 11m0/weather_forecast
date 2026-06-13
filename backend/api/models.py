import requests
from fastapi import APIRouter, HTTPException

from backend.api.schemas import CityName, ForecastHorizon, ModelName
from backend.services.model_service import ModelService


router = APIRouter(
    prefix="/models",
    tags=["models"],
)

model_service = ModelService()
dataset_service = model_service.dataset_service


def train_and_save_model(city: str, model_name: str) -> dict:
    """
    Обучить модель через единый сервис и сохранить артефакты.

    Args:
        city: Идентификатор города.
        model_name: Название модели.

    Returns:
        Метрики и пути к сохранённым файлам.
    """
    return model_service.train(city, model_name)


def build_train_response(city: str, model_name: str) -> dict:
    """
    Выполнить обучение и сформировать стандартный ответ API.

    Args:
        city: Идентификатор города.
        model_name: Название модели.

    Returns:
        Стандартный результат обучения.
    """
    try:
        result = train_and_save_model(city, model_name)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Датасет города '{city}' не найден.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "city": city,
        "model": model_name,
        **result,
    }


def build_forecast_response(
    city: str,
    model_name: str,
    horizon: int,
) -> dict:
    """
    Выполнить прогноз и сформировать стандартный ответ API.

    Args:
        city: Идентификатор города.
        model_name: Название модели.
        horizon: Горизонт прогноза в днях.

    Returns:
        Стандартный результат прогнозирования.
    """
    try:
        result = model_service.forecast(city, model_name, horizon)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Датасет или модель '{model_name}' для города "
                f"'{city}' не найдены. Сначала обновите данные "
                "и обучите модель."
            ),
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "city": city,
        "model": model_name,
        "horizon": horizon,
        **result,
    }


@router.post("/train")
def train_model(city: CityName, model_name: ModelName) -> dict:
    """
    Обучить выбранную модель на сохранённом датасете.

    Args:
        city: Идентификатор города.
        model_name: Название модели.

    Returns:
        Метрики обучения и пути к артефактам.
    """
    return build_train_response(city.value, model_name.value)


@router.post("/update-and-train")
def update_data_and_train(
    city: CityName,
    model_name: ModelName,
) -> dict:
    """
    Обновить погодные данные и переобучить выбранную модель.

    Args:
        city: Идентификатор города.
        model_name: Название модели.

    Returns:
        Результаты обновления данных и обучения.
    """
    try:
        data_result = model_service.update_data(city.value)
    except (requests.RequestException, KeyError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось обновить погодные данные: {error}",
        ) from error

    training_response = build_train_response(
        city.value,
        model_name.value,
    )
    return {
        **training_response,
        "data": data_result,
    }


@router.get("/forecast")
def forecast_model(
    city: CityName,
    model_name: ModelName,
    horizon: ForecastHorizon = 7,
) -> dict:
    """
    Получить прогноз выбранной модели на срок от 1 до 7 дней.

    Args:
        city: Идентификатор города.
        model_name: Название модели.
        horizon: Горизонт прогноза в днях.

    Returns:
        Сохранённый прогноз выбранной модели.
    """
    return build_forecast_response(
        city.value,
        model_name.value,
        horizon,
    )


@router.get("/comparison")
def compare_forecast_with_actual(
    city: CityName,
    model_name: ModelName,
) -> dict:
    """
    Получить сравнение сохранённых прогнозов с фактической погодой.

    Args:
        city: Идентификатор города.
        model_name: Название модели.

    Returns:
        Совпавшие точки и постфактум метрики.
    """
    comparison = model_service.compare_with_actual(
        city.value,
        model_name.value,
    )
    return {
        "city": city.value,
        "model": model_name.value,
        **comparison,
    }


@router.get("/metrics")
def get_models_metrics(city: CityName) -> dict:
    """
    Получить метрики всех обученных моделей города.

    Args:
        city: Идентификатор города.

    Returns:
        Метрики по названиям моделей.
    """
    return {
        "city": city.value,
        "metrics": model_service.get_metrics(city.value),
    }


@router.get("/list")
def get_models() -> dict:
    """Получить список доступных моделей."""
    return {"models": list(ModelService.MODEL_NAMES)}


@router.post("/regression/train", deprecated=True)
def train_regression_model(city: CityName) -> dict:
    """Обучить regression через устаревший совместимый endpoint."""
    return build_train_response(city.value, "regression")


@router.post("/arima/train", deprecated=True)
def train_arima_model(city: CityName) -> dict:
    """Обучить ARIMA через устаревший совместимый endpoint."""
    return build_train_response(city.value, "arima")


@router.post("/lstm/train", deprecated=True)
def train_lstm_model(city: CityName) -> dict:
    """Обучить LSTM через устаревший совместимый endpoint."""
    return build_train_response(city.value, "lstm")


@router.get("/regression/forecast", deprecated=True)
def forecast_regression_model(
    city: CityName,
    horizon: ForecastHorizon = 7,
) -> dict:
    """Получить regression-прогноз через совместимый endpoint."""
    return build_forecast_response(city.value, "regression", horizon)


@router.get("/arima/forecast", deprecated=True)
def forecast_arima_model(
    city: CityName,
    horizon: ForecastHorizon = 7,
) -> dict:
    """Получить ARIMA-прогноз через совместимый endpoint."""
    return build_forecast_response(city.value, "arima", horizon)


@router.get("/lstm/forecast", deprecated=True)
def forecast_lstm_model(
    city: CityName,
    horizon: ForecastHorizon = 7,
) -> dict:
    """Получить LSTM-прогноз через совместимый endpoint."""
    return build_forecast_response(city.value, "lstm", horizon)


@router.get("/regression/predict", deprecated=True)
def predict_regression_model(city: CityName) -> list[dict]:
    """
    Получить regression-предсказания на историческом датасете.

    Args:
        city: Идентификатор города.

    Returns:
        Фактические и предсказанные значения по датам.
    """
    try:
        return model_service.predict_history(city.value)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=(
                "Датасет или regression-модель не найдены. "
                "Сначала обновите данные и обучите модель."
            ),
        ) from error
