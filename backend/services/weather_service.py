import requests


class WeatherService:
    """
    Сервис для получения погодных данных из Open-Meteo API.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Получить исторические погодные данные.

        Args:
            latitude: Широта населённого пункта.
            longitude: Долгота населённого пункта.
            start_date: Начальная дата в формате YYYY-MM-DD.
            end_date: Конечная дата в формате YYYY-MM-DD.

        Returns:
            Ответ Open-Meteo API в формате JSON.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_mean",
                "temperature_2m_min",
                "temperature_2m_max",
                "precipitation_sum",
                "windspeed_10m_max",
            ],
            "timezone": "auto",
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=20,
        )
        response.raise_for_status()

        return response.json()

    @staticmethod
    def normalize_daily_weather(raw_data: dict, city: str) -> list[dict]:
        """
        Преобразовать ответ Open-Meteo в плоский список погодных записей.

        Args:
            raw_data: Исходный JSON-ответ Open-Meteo.
            city: Идентификатор города.

        Returns:
            Список записей с погодными данными по дням.
        """
        daily = raw_data["daily"]

        return [
            {
                "date": daily["time"][index],
                "city": city,
                "temp_mean": daily["temperature_2m_mean"][index],
                "temp_min": daily["temperature_2m_min"][index],
                "temp_max": daily["temperature_2m_max"][index],
                "precipitation": daily["precipitation_sum"][index],
                "wind_max": daily["windspeed_10m_max"][index],
            }
            for index in range(len(daily["time"]))
        ]
