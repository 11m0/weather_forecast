from backend.ml.arima_model import ArimaModel
from backend.ml.lstm_model import LstmModel
from backend.ml.regression_model import RegressionModel


class ModelFactory:
    """
    Фабрика для создания моделей прогнозирования.
    """

    MODELS = {
        "regression": RegressionModel,
        "arima": ArimaModel,
        "lstm": LstmModel,
    }

    @classmethod
    def create(cls, model_name: str):
        """
        Создать экземпляр модели.

        Args:
            model_name: Название модели.

        Returns:
            Экземпляр модели.

        Raises:
            ValueError: Если модель не поддерживается.
        """
        model_class = cls.MODELS.get(model_name)

        if model_class is None:
            raise ValueError(
                f"Unsupported model '{model_name}'."
            )

        return model_class()
