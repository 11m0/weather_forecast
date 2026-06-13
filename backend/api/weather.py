from datetime import date

from fastapi import APIRouter, HTTPException

from backend.api.schemas import CityName
from backend.config import CITIES
from backend.ml.feature_builder import FeatureBuilder
from backend.services.dataset_service import DatasetService
from backend.services.weather_service import WeatherService


router = APIRouter(
    prefix="/weather",
    tags=["weather"],
)

weather_service = WeatherService()
dataset_service = DatasetService()
feature_builder = FeatureBuilder()


@router.get("/history")
def get_weather_history(
    city: CityName,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Получить исторические погодные данные для города.

    Args:
        city: Идентификатор города из конфигурации.
        start_date: Начальная дата в формате YYYY-MM-DD.
        end_date: Конечная дата в формате YYYY-MM-DD.

    Returns:
        Исторические погодные данные.
    """
    if start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="Начальная дата не может быть позже конечной.",
        )

    city_value = city.value
    city_data = CITIES[city_value]

    raw_data = weather_service.get_historical_weather(
        latitude=city_data["latitude"],
        longitude=city_data["longitude"],
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    normalized_data = weather_service.normalize_daily_weather(
        raw_data=raw_data,
        city=city_value,
    )

    return normalized_data


@router.post("/download")
def download_weather_data(
    city: CityName,
    start_date: date,
    end_date: date,
) -> dict:
    """
    Скачать, нормализовать и сохранить погодные данные.

    Args:
        city: Идентификатор города.
        start_date: Начальная дата в формате YYYY-MM-DD.
        end_date: Конечная дата в формате YYYY-MM-DD.

    Returns:
        Информация о сохранённом датасете.
    """
    if start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="Начальная дата не может быть позже конечной.",
        )

    city_value = city.value
    city_data = CITIES[city_value]

    raw_data = weather_service.get_historical_weather(
        latitude=city_data["latitude"],
        longitude=city_data["longitude"],
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    records = weather_service.normalize_daily_weather(
        raw_data=raw_data,
        city=city_value,
    )

    file_path = dataset_service.save_weather_data(
        records=records,
        city=city_value,
    )

    return {
        "city": city_value,
        "records_count": len(records),
        "file_path": str(file_path),
    }


@router.get("/dataset")
def get_saved_weather_dataset(city: CityName) -> list[dict]:
    """
    Получить сохранённый локальный погодный датасет.

    Args:
        city: Идентификатор города.

    Returns:
        Список записей из локального CSV-файла.
    """
    try:
        dataframe = dataset_service.load_weather_data(city=city.value)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return dataframe.to_dict(orient="records")


@router.get("/features")
def get_weather_features(city: CityName) -> list[dict]:
    """
    Получить датасет с признаками для обучения модели.

    Args:
        city: Идентификатор города.

    Returns:
        Список записей с добавленными признаками.
    """
    try:
        dataframe = dataset_service.load_weather_data(city=city.value)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    features = feature_builder.build_features(dataframe)

    return features.to_dict(orient="records")
