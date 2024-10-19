import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from unittest.mock import MagicMock

from train_lr import Dataset, get_train_test_data, train_model, validate_model


@pytest.fixture
def mock_data():
    return np.array([[1, 2, 3, 4], [5, 6, 7, 8]])


@pytest.fixture
def mock_target():
    return np.array([0, 1])


def test_dataset_class(mock_data, mock_target):
    dataset = Dataset(data=mock_data, target=mock_target)
    assert np.array_equal(dataset.data, mock_data)
    assert np.array_equal(dataset.target, mock_target)

    head_data, head_target = dataset.head(1)
    assert np.array_equal(head_data, np.array([[1, 2, 3, 4]]))
    assert np.array_equal(head_target, np.array([0]))


def test_get_train_test_data(mock_data, mock_target):
    train_dataset, test_dataset = get_train_test_data(mock_data, mock_target)

    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)
    assert np.array_equal(train_dataset.data, np.array([[1, 2, 3, 4]]))
    assert np.array_equal(test_dataset.data, np.array([[5, 6, 7, 8]]))


def test_train_model(mock_data, mock_target):
    dataset = Dataset(data=mock_data, target=mock_target)
    model = train_model(dataset)
    assert isinstance(model, LogisticRegression)


def test_validate_model_with_mocked_predict(mock_data, mock_target):
    mock_model = MagicMock(spec=LogisticRegression)
    mock_model.predict.return_value = np.array([0, 1])

    dataset = Dataset(data=mock_data, target=mock_target)

    metrics = validate_model(mock_model, dataset)

    mock_model.predict.assert_called_once_with(mock_data)
    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert metrics["accuracy"] == 1.0
