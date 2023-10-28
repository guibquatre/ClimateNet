import os
import logging
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)


class DataLoader:
    @staticmethod
    def safe_load_csv_dataset(file_path: str):
        try:
            data = pd.read_csv(file_path)
            labels = data.get('Label')
            features = data.drop('Label', axis=1) if labels is not None else data
            return features, labels
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            return None, None


class ModelSaver:
    @staticmethod
    def save_model(_model, path: str):
        try:
            with open(path, 'wb') as f:
                pickle.dump(_model, f)
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")


class SubmissionSaver:
    @staticmethod
    def save_submission(_predictions):
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            submission_file_path = f"./submission_{timestamp}.csv"
            pd.DataFrame({'SNo': range(1, len(_predictions) + 1),
                          'Label': _predictions}).to_csv(submission_file_path, index=False)
        except Exception as e:
            logging.error(f"Error saving submission: {str(e)}")


class SimpleDummyClassifier:
    def __init__(self):
        self.unique_labels = None

    def fit(self, not_used_by_purpose_X: np.array, y: np.array) -> None:
        self.unique_labels = np.unique(y)

    def predict(self, not_used_by_purpose_X: np.array) -> np.array:
        num_samples = not_used_by_purpose_X.shape[0]
        return np.random.choice(self.unique_labels, size=num_samples)


class SimpleLogisticRegression:
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)  # Clip values to avoid overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.array, y: np.array) -> None:
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            _predictions = self.sigmoid(linear_model)
            gradient_weights = (1 / num_samples) * np.dot(X.T, (_predictions - y))
            gradient_bias = (1 / num_samples) * np.sum(_predictions - y)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X: np.array) -> np.array:
        linear_model = np.dot(X, self.weights) + self.bias
        _predictions = self.sigmoid(linear_model)
        return np.round(_predictions).astype(int)


if __name__ == '__main__':
    data_path = "/home/gui/INF6390/competition/ClimateNet/example/classification-of-extreme-weather-events-udem"
    file_names = {'train': 'train.csv', 'test': 'test.csv'}

    loader = DataLoader()
    train_data, train_labels = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['train']))
    test_data, _ = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['test']))

    if train_data is None or test_data is None:
        logging.error("Data loading failed. Cannot proceed.")
        exit(1)

    train_data, val_data, train_labels, val_labels \
        = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    models = [
        ('Dummy', SimpleDummyClassifier()),
        ('SimpleLogisticRegression', SimpleLogisticRegression())
    ]

    best_f1 = 0
    best_model = None

    for name, model in models:
        model.fit(train_data, train_labels)
        predictions = model.predict(val_data)
        report = classification_report(val_labels, predictions, output_dict=True, zero_division=1)
        f1 = report['weighted avg']['f1-score']

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    SubmissionSaver.save_submission(best_model.predict(test_data))
    ModelSaver.save_model(best_model, "best_model.pkl")
