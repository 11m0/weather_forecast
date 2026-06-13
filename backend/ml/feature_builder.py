import pandas as pd


class FeatureBuilder:
    """
    Класс для построения признаков из погодного временного ряда.
    """

    @staticmethod
    def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Построить признаки для обучения модели.

        Args:
            dataframe: Исходный датасет с погодными данными.

        Returns:
            Датасет с добавленными признаками.
        """
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])

        dataframe = dataframe.sort_values("date")

        dataframe["month"] = dataframe["date"].dt.month
        dataframe["day_of_year"] = dataframe["date"].dt.dayofyear
        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek

        dataframe["temp_lag_1"] = dataframe["temp_mean"].shift(1)
        dataframe["temp_lag_2"] = dataframe["temp_mean"].shift(2)
        dataframe["temp_lag_3"] = dataframe["temp_mean"].shift(3)
        dataframe["temp_lag_7"] = dataframe["temp_mean"].shift(7)

        dataframe["temp_rolling_3"] = dataframe["temp_mean"].rolling(window=3).mean()
        dataframe["temp_rolling_7"] = dataframe["temp_mean"].rolling(window=7).mean()

        dataframe = dataframe.dropna().reset_index(drop=True)

        return dataframe

    @staticmethod
    def build_next_day_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Построить признаки для прогноза на следующий день.

        Args:
            dataframe: Исторический датасет с погодными данными.

        Returns:
            DataFrame с одной строкой признаков для следующего дня.
        """
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = dataframe.sort_values("date").reset_index(drop=True)

        last_date = dataframe["date"].max()
        next_date = last_date + pd.Timedelta(days=1)

        row = {
            "date": next_date,
            "month": next_date.month,
            "day_of_year": next_date.dayofyear,
            "day_of_week": next_date.dayofweek,
            "temp_lag_1": dataframe["temp_mean"].iloc[-1],
            "temp_lag_2": dataframe["temp_mean"].iloc[-2],
            "temp_lag_3": dataframe["temp_mean"].iloc[-3],
            "temp_lag_7": dataframe["temp_mean"].iloc[-7],
            "temp_rolling_3": dataframe["temp_mean"].tail(3).mean(),
            "temp_rolling_7": dataframe["temp_mean"].tail(7).mean(),
        }

        return pd.DataFrame([row])
