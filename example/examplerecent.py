import os
import csv
import pandas as pd
import logging
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from typing import Tuple, Union, Dict, List

# Initialize logging
logging.basicConfig(level=logging.INFO)

class DataLoader:
    @staticmethod
    def safe_load_csv_dataset(file_path: str) -> Tuple[Union[pd.DataFrame, None], Union[pd.Series, None]]:
        try:
            data = pd.read_csv(file_path)
            labels = data['Label'] if 'Label' in data.columns else None
            features = data.drop('Label', axis=1) if labels is not None else data
            return features, labels
        except Exception as e:
            logging.error("Error loading dataset:: %s", str(e))
            return None, None

    def load_datasets(self, _data_path: str, _file_names: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        train_data, train_labels = self.safe_load_csv_dataset(os.path.join(_data_path, _file_names['train']))
        test_data, _ = self.safe_load_csv_dataset(os.path.join(_data_path, _file_names['test']))
        return train_data, train_labels, test_data

class ModelSaver:
    @staticmethod
    def save_model(model, path: str) -> None:
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logging.error("Error saving model: %s", str(e))

    @staticmethod
    def load_model(path: str):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logging.error("Error loading model: %s", str(e))
            return None

class SubmissionSaver:
    @staticmethod
    def save_submission(_predictions):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        submission_file_path = "./submission_{}.csv".format(timestamp)
        try:
            with open(submission_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['SNo', 'Label'])
                for idx, label in enumerate(_predictions, 1):
                    writer.writerow([idx, label])
        except Exception as e:
            logging.error("Error saving the submission: %s", str(e))

class SimpleDummyClassifier:
    def fit(self, X: np.array, y: np.array) -> None:
        self.unique_labels = np.unique(y)

    def predict(self, X: np.array) -> np.array:
        num_samples = X.shape[0]
        return np.random.choice(self.unique_labels, size=num_samples)

class SimpleLogisticRegression:
    class SimpleLogisticRegression:
        def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
            self.learning_rate = learning_rate
            self.num_iterations = num_iterations
            self.weights = None
            self.bias = 0
            self.training_set = {'data': None, 'labels': None}
            self.inference_set = {'data': None}

        @staticmethod
        def safe_load_csv_dataset(file_path: str) -> Tuple[Union[pd.DataFrame, None], Union[pd.Series, None]]:
            try:
                data = pd.read_csv(file_path)
                labels = data['Label'] if 'Label' in data.columns else None
                features = data.drop('Label', axis=1) if labels is not None else data
                return features, labels
            except Exception as e:
                logging.error("Error loading dataset:: %s", str(e))
                return None, None

        @staticmethod
        def sigmoid(z: np.array) -> np.array:
            return 1 / (1 + np.exp(-z))

        def train(self, X: np.array, y: np.array) -> None:
            num_samples, num_features = X.shape
            self.weights = np.zeros(num_features)
            for _ in range(self.num_iterations):
                linear_model = np.dot(X, self.weights) + self.bias
                _predictions = self.sigmoid(linear_model)
                gradient_weights = (1 / num_samples) * np.dot(X.T, (_predictions - y))
                gradient_bias = (1 / num_samples) * np.sum(_predictions - y)
                self.weights -= self.learning_rate * gradient_weights
                self.bias -= self.learning_rate * gradient_bias

        def predict(self, X: np.array) -> List[int]:
            linear_model = np.dot(X, self.weights) + self.bias
            _predictions = self.sigmoid(linear_model)
            return [1 if i > 0.5 else 0 for i in _predictions]

        def load_datasets(self, path: str, _file_names: Dict[str, str]) -> None:
            self.training_set['data'], self.training_set['labels'] = self.safe_load_csv_dataset(
                os.path.join(path, _file_names['train']))
            self.inference_set['data'], _ = self.safe_load_csv_dataset(os.path.join(path, _file_names['test']))

        def evaluate_model(self, model):
            _predictions = model.predict(self.inference_set['data'])

            # Check if inference labels are present
            if self.inference_set.get('labels') is not None:
                report = classification_report(self.inference_set['labels'], _predictions, output_dict=True,
                                               zero_division=1)
                return report['weighted avg']['f1-score']
            else:
                logging.info("Inference set labels are not available. Returning None.")
                return None

        def train_and_evaluate(self):
            if self.training_set['data'] is not None:
                X_train, X_val, y_train, y_val = train_test_split(self.training_set['data'],
                                                                  self.training_set['labels'],
                                                                  test_size=0.2, random_state=42)

                baseline_models = [
                    ('Dummy', SimpleDummyClassifier),
                    ('SimpleLogisticRegression', SimpleLogisticRegression())
                ]

                best_f1 = 0.0
                best_model_name = ""
                _best_model = None

                for name, model in baseline_models:
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                    elif hasattr(model, 'train'):
                        model.train(X_train, y_train)

                    f1 = self.evaluate_model(model)

                    # Check if F1 score was returned
                    if f1 is not None and f1 > best_f1:
                        best_f1 = f1
                        best_model_name = name
                        _best_model = model

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                model_directory = "./saved_models/"
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)

                best_model_filename = "{}_{}_best_model.pkl".format(best_model_name, timestamp)
                best_model_path = os.path.join(model_directory, best_model_filename)

                with open(best_model_path, 'wb') as f:
                    pickle.dump(_best_model, f)

                return _best_model
            else:
                logging.error("Training data is missing. Cannot proceed.")
                return None

        @staticmethod
        def save_submission(_predictions):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            submission_file_path = "./submission_{}.csv".format(timestamp)
            try:
                with open(submission_file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['SNo', 'Label'])
                    for idx, label in enumerate(_predictions, 1):
                        writer.writerow([idx, label])
            except Exception as e:
                logging.error("Error saving the submission: %s", str(e))

class BaselineModelsPipeline:
    def __init__(self, data_path: str, file_names: Dict[str, str]):
        self.models = [
            ('Dummy', SimpleDummyClassifier()),
            ('SimpleLogisticRegression', SimpleLogisticRegression())
        ]
        self.data_path = data_path
        self.file_names = file_names
        self.pipeline = SimpleLogisticRegression()

    def run(self):
        loader = DataLoader()
        train_data, train_labels, test_data = loader.load_datasets(self.data_path, self.file_names)
        best_model = self.pipeline.train_and_evaluate(train_data, train_labels, self.models)
        if best_model:
            predictions = best_model.predict(test_data)
            SubmissionSaver.save_submission(predictions)
            ModelSaver.save_model(best_model, "best_model.pkl")

if __name__ == '__main__':
    data_path = "/home/gui/INF6390/competition/ClimateNet/example/classification-of-extreme-weather-events-udem/"
    file_names = {'train': 'train.csv', 'test': 'test.csv'}

    pipeline = BaselineModelsPipeline(data_path, file_names)
    pipeline.run()
