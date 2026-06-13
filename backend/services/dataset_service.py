from pathlib import Path

import pandas as pd


class DatasetService:
    """
    Сервис для сохранения и загрузки локальных датасетов.
    """

    DATA_DIR = Path("data/processed")

    def save_weather_data(self, records: list[dict], city: str) -> Path:
        """
        Сохранить погодные записи в CSV-файл.

        Args:
            records: Список погодных записей.
            city: Идентификатор города.

        Returns:
            Путь к сохранённому CSV-файлу.
        """
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        file_path = self.DATA_DIR / f"{city}_weather.csv"

        dataframe = pd.DataFrame(records)
        dataframe.to_csv(file_path, index=False)

        return file_path

    def merge_weather_data(
        self,
        records: list[dict],
        city: str,
    ) -> tuple[Path, int, int]:
        """
        Объединить новые погодные записи с локальным датасетом.

        Args:
            records: Новые погодные записи.
            city: Идентификатор города.

        Returns:
            Путь к датасету, число новых дат и общее число записей.

        Raises:
            ValueError: Если список погодных записей пуст.
        """
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path = self.DATA_DIR / f"{city}_weather.csv"
        new_data = pd.DataFrame(records)

        if file_path.exists():
            current_data = pd.read_csv(file_path)
            current_dates = set(current_data["date"].astype(str))
            dataframe = pd.concat([current_data, new_data], ignore_index=True)
        else:
            current_dates = set()
            dataframe = new_data

        if dataframe.empty:
            raise ValueError("Weather API returned no records.")

        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = (
            dataframe
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        saved_dates = set(dataframe["date"].dt.strftime("%Y-%m-%d"))
        added_rows = len(saved_dates - current_dates)
        dataframe["date"] = dataframe["date"].dt.strftime("%Y-%m-%d")
        dataframe.to_csv(file_path, index=False)

        return file_path, added_rows, len(dataframe)

    def load_weather_data(self, city: str) -> pd.DataFrame:
        """
        Загрузить датасет города из CSV.

        Args:
            city: Идентификатор города.

        Returns:
            DataFrame с погодными данными.

        Raises:
            FileNotFoundError: Если файл не существует.
        """
        file_path = self.DATA_DIR / f"{city}_weather.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset for city '{city}' not found."
            )

        return pd.read_csv(file_path)
