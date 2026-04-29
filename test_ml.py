import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, inference, compute_model_metrics





def test_train_model_returns_logistic_regression():
    """
    Test that train_model returns a LogisticRegression model
    """
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)



def test_inference_output_length():
    """
    Test that inference returns correct number of predictions
    """
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(X)



def test_computer_model_metrics_types():
    """
    Test that computer_model_metrics returns floats.
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
