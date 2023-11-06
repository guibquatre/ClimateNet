import os
import csv
import logging
import joblib
import numpy as np
from datetime import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import xgboost as xgb

# Keep your logging setup and DataLoader class
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


class Preprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    # Compute the values along each column, independently
    def fit(self, X):
        # Compute and store the mean of each feature/column in the dataset
        self.mean = np.mean(X, axis=0)
        # Compute and store the standard deviation of each feature/column in the dataset
        self.std = np.std(X, axis=0)
        # Compute and store the minimum value of each feature/column in the dataset
        self.min = np.min(X, axis=0)
        # Compute and store the maximum value of each feature/column in the dataset
        self.max = np.max(X, axis=0)

    # Method to apply either standardization or normalization to the dataset
    def transform(self, X, method='standardize'):
        # Check the method argument to determine the transformation to apply
        if method == 'standardize':
            # Call the standardize method (not defined in provided code) to standardize the dataset
            return self.standardize(X)
        elif method == 'normalize':
            # Call the normalize method to normalize the dataset
            return self.normalize(X)
        else:
            # If an unknown method is provided, raise a ValueError with a descriptive message
            raise ValueError(f"Unknown method: {method}")

    def standardize(self, X):
        # Standardization formula: (X - mean) / std
        # Ensure to add a small value to the denominator to avoid division by zero
        return (X - self.mean) / (self.std + 1e-8)

    def normalize(self, X):
        # Normalization formula: (X - min) / (max - min)
        # Ensure to add a small value to the denominator to avoid division by zero
        return (X - self.min) / (self.max - self.min + 1e-8)


class SimpleDummyClassifier:
    def __init__(self):
        self.unique_labels = None

    def fit(self, _: np.array, y: np.array) -> None:
        self.unique_labels = np.unique(y)

    def predict(self, _: np.array) -> np.array:
        num_samples = _.shape[0]
        return np.random.choice(self.unique_labels, size=num_samples)


class SoftLogisticRegression:
    # initialize the hyperparameters and sets up the initial values of weights and bias.
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_strength=0.1):
        self.learning_rate = learning_rate  # step size used during optimization to find the minimum of loss function.
        self.num_iterations = num_iterations  # number of steps the optimizer will take to minimize loss function.
        self.regularization_strength = regularization_strength  # Controls regularization strength, prevent overfitting.
        self.weights = None  # Placeholder for the weights vector that will be learned from the data.
        self.bias = None  # Placeholder for the bias term that will be learned from the data.

    @staticmethod
    def softmax(z):
        # For numerical stability, subtract the maximum value of z for each sample.
        z -= np.max(z, axis=1, keepdims=True)
        # Compute the exponential of z to get unnormalized probabilities.
        exp_z = np.exp(z)
        # Sum the unnormalized probabilities for each sample to normalize them.
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        # Divide each unnormalized probability by the sum to get the normalized probabilities.
        return exp_z / sum_exp_z

    def fit(self, X, y):
        y = y.astype(int)  # Convert the labels to integer type in case they are not for indexing
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))  # Get the number of unique labels, which equals the number of classes.
        self.weights = np.zeros((num_features, num_classes))  # Initialize the weights matrix with zeros.
        self.bias = np.zeros(num_classes)  # Initialize the bias vector with zeros.
        y_one_hot = np.eye(num_classes)[y]
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(linear_model)
            error = probabilities - y_one_hot
            gradient_weights = (1 / num_samples) * np.dot(X.T, error) + self.regularization_strength * self.weights
            gradient_bias = (1 / num_samples) * np.sum(error, axis=0)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(linear_model)
        return np.argmax(probabilities, axis=1)

def calculate_metrics(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    # Step 6: Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def classification_report_custom(y_true, y_pred):

    labels = np.unique(y_true)

    # Step 2: Iterate Over Each Unique Label
    for label in labels:
        precision, recall, f1 = calculate_metrics(y_true, y_pred, label)
        # Step 3: Display Metrics for Each Label
        print(f'Label: {label}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')


class ClimateAnalysisPipeline:
    def __init__(self):
        logging.info("Initializing ClimateAnalysisPipeline")
        self.training_set = {'data': None, 'labels': None}
        self.inference_set = {'data': None, 'labels': None}

    def load_datasets(self, _path, _file_names):
        loader = DataLoader()
        self.training_set['data'], self.training_set['labels'] = \
            loader.safe_load_csv_dataset(os.path.join(_path, _file_names['train']), True)
        self.inference_set['data'], _ = \
            loader.safe_load_csv_dataset(os.path.join(_path, _file_names['test']), False)

    def train_and_evaluate(self):
        logging.info("Starting model training and evaluation")
        if self.training_set['data'] is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                self.training_set['data'],
                self.training_set['labels'],
                test_size=0.2, random_state=42
            )

            # Adding XGBoost to the model list
            xg_reg = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
                                       max_depth=5, alpha=10, n_estimators=10)
            # Run gridsearch.py before this line
            grid_search_rf = joblib.load('remise/grid_search_rf_final.joblib')
            # Get the best estimator from the GridSearch
            best_rf = grid_search_rf.best_estimator_

            # Initialize a list to hold tuples of model names and their instances.
            baseline_models = [
                # Classifier likely used for establishing a basic benchmark.
                # It does not learn from the data and predicts using a simple heuristic.
                ('SimpleDummy', SimpleDummyClassifier()),
                # A variant of logistic regression tailored to provide a probabilistic
                # output, implying a focus on modeling the uncertainty in predictions.
                ('SoftLogisticRegression', SoftLogisticRegression()),
                # A classifier that disregards input features and predicts uniformly at random,
                # ensuring no class is favored; primarily for baseline comparison.
                ('Dummy', DummyClassifier(strategy="uniform")),
                # Implements a linear model with stochastic gradient descent learning, with
                # balanced class weights to correct for any imbalance in the dataset.
                ('SGD', SGDClassifier(class_weight='balanced')),
                # A robust classifier that finds an optimal hyperplane for class separation,
                # applying balanced weights to account for unequal class representation.
                ('SVC', SVC(class_weight='balanced')),
                # A pre-optimized RandomForest model, indicating prior hyperparameter tuning
                # or feature selection to maximize the model's predictive accuracy.
                ('RandomForest_best_rf', best_rf),
                # An ensemble of decision trees with class weights balanced to improve the
                # decision-making process where data is skewed across different classes.
                ('RandomForest', RandomForestClassifier(class_weight='balanced')),
                # A logistic regression model configured for a high number of iterations and
                # balanced class weights, suitable for complex or imbalanced datasets.
                ('LogisticRegression', LogisticRegression(max_iter=1000, class_weight='balanced')),
                # An advanced gradient boosting framework that is designed for speed and
                # performance, often outperforming other models on structured datasets.
                ('XGBoost', xg_reg)
            ]

            best_f1 = 0.0
            best_model_name = ""
            _best_model = None

            for name, model in baseline_models:
                model.fit(X_train, y_train)
                _predictions = model.predict(X_val)
                print(f'Performance of {name}:')
                classification_report_custom(y_val, _predictions)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                    report = classification_report(y_val, _predictions, output_dict=True, zero_division=0)
                f1 = report['weighted avg']['f1-score']
                logging.info("Model {} F1-score: {:.4f}".format(name, f1))

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = name
                    _best_model = model

            logging.info("Best model is {} with F1-score of {:.4f}".format(best_model_name, best_f1))
            ModelSaver.save_model(_best_model)
            return _best_model
        else:
            logging.error("Training data is missing. Cannot proceed.")
            return None


def main():
    data_path = "classification-of-extreme-weather-events-udem"
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

    # Create an instance of ClimateAnalysisPipeline
    pipeline = ClimateAnalysisPipeline()

    # Set the training and inference datasets
    pipeline.training_set['data'], pipeline.training_set['labels'] = train_data, train_labels
    pipeline.inference_set['data'] = test_data

    # Call train_and_evaluate
    best_model = pipeline.train_and_evaluate()

    if best_model:
        test_predictions = best_model.predict(test_data)
        SubmissionSaver.save_submission(test_predictions)
    else:
        logging.error("No best model, sorry")


if __name__ == '__main__':
    main()
