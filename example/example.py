import os
import logging
import joblib
import csv
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)


class DataLoader:
    @staticmethod
    def safe_load_csv_dataset(file_path: str, is_train=True):
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                data = np.array([row for row in reader], dtype=float)

            if is_train:
                # If it's the training data, assume labels are in the last column
                labels = data[:, -1]
                features = data[:, :-1]
            else:
                # If it's the testing data, there are no labels
                labels = None
                features = data

            return features, labels

        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            return None, None
        except ValueError as e:
            logging.error(f"Value error: {str(e)}")
            return None, None


class SubmissionSaver:
    @staticmethod
    def save_submission(predictions, file_path='submission.csv'):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SNo', 'Label'])
            for i, label in enumerate(predictions, 1):
                writer.writerow([i, label])
        logging.info(f'Submission saved to {file_path}')


class Preprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X, method='standardize'):
        if method == 'standardize':
            return self.standardize(X)
        elif method == 'normalize':
            return self.normalize(X)
        else:
            raise ValueError(f"Unknown method: {method}")

    def standardize(self, X):
        return (X - self.mean) / self.std

    def normalize(self, X):
        return (X - self.min) / (self.max - self.min)


class ModelSaver:
    @staticmethod
    def save_model(_model):
        try:
            os.makedirs("saved_models", exist_ok=True)
            class_name = _model.__class__.__name__
            timestamp = datetime.now().strftime("%d_%H_%M_%S")
            filename = f"saved_models/{class_name}_{timestamp}.joblib"
            joblib.dump(_model, filename)
            logging.info(f"Model saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")


class SimpleDummyClassifier:
    def __init__(self):
        self.unique_labels = None

    def fit(self, _: np.array, y: np.array) -> None:
        self.unique_labels = np.unique(y)

    def predict(self, not_used_by_purpose_X: np.array) -> np.array:
        num_samples = not_used_by_purpose_X.shape[0]
        return np.random.choice(self.unique_labels, size=num_samples)


class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_strength=0.1, class_weights=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength
        self.class_weights = class_weights  # New class_weights parameter
        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        y = y.reshape(-1, 1)  # Ensure y is a column vector

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model).reshape(-1, 1)  # Ensure predictions is a column vector

            # Get class weights
            if self.class_weights is not None:
                weights_vector = np.array([self.class_weights[label] for label in y.flatten()]).reshape(-1, 1)
            else:
                weights_vector = np.ones_like(y)

            # Compute gradients (with L2 regularization and class weights)
            gradient_weights = (1 / num_samples) * np.dot(X.T, weights_vector * (
                        predictions - y)).flatten() + self.regularization_strength * self.weights
            gradient_bias = (1 / num_samples) * np.sum(weights_vector * (predictions - y))

            # Update weights and bias
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return np.round(predictions).astype(int)  # Threshold at 0.5


def train_test_split(X: np.array, y: np.array, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def calculate_metrics(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def classification_report(y_true, y_pred):
    labels = np.unique(y_true)
    for label in labels:
        precision, recall, f1 = calculate_metrics(y_true, y_pred, label)
        print(f'Label: {label}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')


def main():
    data_path = "/Users/spokboud/INF6390/competition/ClimateNet/example/classification-of-extreme-weather-events-udem"
    file_names = {'train': 'train.csv', 'test': 'test.csv'}

    loader = DataLoader()
    train_data, train_labels = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['train']), True)
    test_data, _ = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['test']), False)

    if train_data is None or test_data is None:
        logging.error("Data loading failed. Cannot proceed.")
        exit(1)

    preprocessor = Preprocessor()
    preprocessor.fit(train_data)  # Compute statistics based on the training data
    # Standardize the training data
    train_data = preprocessor.transform(train_data, method='standardize')
    # Standardize the test data using the same statistics
    test_data = preprocessor.transform(test_data, method='standardize')

    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels)

    models = [
        ('Dummy', SimpleDummyClassifier()),
        ('SimpleLogisticRegression', SimpleLogisticRegression())
    ]

    best_f1 = 0
    best_model = None

    for name, model in models:
        model.fit(train_data, train_labels)
        predictions = model.predict(val_data)
        print(f'Performance of {name}:')
        classification_report(val_labels, predictions)

        f1_scores = [calculate_metrics(val_labels, predictions, label)[2] for label in np.unique(val_labels)]
        f1 = np.mean(f1_scores)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    ModelSaver.save_model(best_model)

    test_predictions = best_model.predict(test_data)
    SubmissionSaver.save_submission(test_predictions)


if __name__ == '__main__':
    main()
