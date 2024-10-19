from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from src.logging_util import get_logger
import numpy as np

logger = get_logger(__name__)


@dataclass
class Dataset:
    data: np.ndarray
    target: np.ndarray

    def head(self, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
        return self.data[:n], self.target[:n]


def get_iris_data() -> tuple[np.ndarray, np.ndarray]:
    logger.info("Start get iris data.")
    iris = load_iris()
    logger.info(f"Finishded get iris data. {type(iris)=}")
    return iris.data, iris.target


def get_train_test_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> tuple[Dataset, Dataset]:
    logger.info(f"Start get train test data. {X.shape=}, {y.shape=}, {test_size=}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    train_dataset, test_dataset = (
        Dataset(data=X_train, target=y_train),
        Dataset(data=X_test, target=y_test),
    )
    logger.info(
        f"Finished get train test data. {train_dataset.head()=}, {test_dataset.head()=}"
    )
    return train_dataset, test_dataset


def train_model(train_dataset: Dataset) -> LogisticRegression:
    logger.info(f"Start train model. {train_dataset.head()=}")
    model = LogisticRegression(random_state=42)
    model.fit(train_dataset.data, train_dataset.target)
    logger.info(f"Finished train model. {model.__dict__=}")
    return model


def validate_model(model: LogisticRegression, dataset: Dataset) -> dict[str, float]:
    logger.info(f"Start validate model. {model=}, {dataset.head()=}")
    prediction = model.predict(dataset.data)

    accuracy = accuracy_score(y_true=dataset.target, y_pred=prediction)

    logger.info(f"Finished validate model. {accuracy=}")
    return {"accuracy": accuracy}


def main():
    logger.info("Start training logistic regression")
    data, target = get_iris_data()
    train_dataset, test_dataset = get_train_test_data(X=data, y=target)
    model = train_model(train_dataset=train_dataset)
    train_metrics = validate_model(model=model, dataset=train_dataset)
    test_metrics = validate_model(model=model, dataset=test_dataset)

    logger.info(
        f"Finished training logistic regression. {train_metrics=}, {test_metrics}"
    )


if __name__ == "__main__":
    main()
