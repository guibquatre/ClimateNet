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
            class_name = _model.__class__.__name__
            timestamp = datetime.now().strftime("%d_%H_%M_%S")
            filename = f"{class_name}_{timestamp}.joblib"
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
        # Compute and store the standard deviation for each feature in the dataset 'X'.
        # Standard deviation is a measure of variability that indicates how much, on average, each
        # value in the dataset deviates from the mean.
        # Standard deviation for outlier detection.
        # The use of 'axis=0' ensures that the computation is done column-wise, meaning that
        # the variability low and high variability is assessed for each feature independently.
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
        # The Z-score, or standard score, is a statistical measure that represents the number
        # of standard deviations a data point is from the mean of the dataset.
        # Z = (x - μ) / σ, where
        # 'x' is the value,
        # 'μ' is the mean of the dataset,
        # 'σ' is the standard deviation.
        # Allows each feature to contribute equally to the analysis
        # Z-scores make it possible to detect outliers in data
        return (X - self.mean) / (self.std + 1e-8)  # avoid division by zero

    def normalize(self, X):
        # Min-Max normalization to scale the dataset X to a 0-1 range.
        # Each feature value X is reduced by the minimum value of the feature, then
        # divided by the range of the feature values (max - min).
        return (X - self.min) / (self.max - self.min + 1e-8)  # Avoid division by zero


class SimpleDummyClassifier:
    def __init__(self):
        self.unique_labels = None

    def fit(self, _: np.array, y: np.array) -> None:
        self.unique_labels = np.unique(y)

    def predict(self, _: np.array) -> np.array:
        num_samples = _.shape[0]
        return np.random.choice(self.unique_labels, size=num_samples)


class SoftLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_strength=0.1):
        # The 'learning_rate' is a hyperparameter that controls the amount by which the weights
        # of our machine learning model is updated during training for gradient descent-based
        # optimization algorithms. A smaller learning rate requires more training iterations through the dataset.
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations  # number of steps the optimizer will take to minimize loss function.
        self.regularization_strength = regularization_strength  # Controls regularization strength, prevent overfitting
        self.weights = None
        self.bias = None

    # Softmax function: Converts raw scores (z) to probabilities for each class.
    @staticmethod
    def softmax(z):
        # Subtract the maximum value of z from each set to prevent potential numerical overflow.
        # Numerical overflow can happen when exponentiating large numbers during operations.
        z -= np.max(z, axis=1, keepdims=True)
        # normalize (z) into a probability distribution. exponentiated,
        # positive values suitable for creating a probability distribution.
        exp_z = np.exp(z)
        # Sum the unnormalized probabilities for each sample to normalize them.
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        # Divide each unnormalized probability by the sum to get the normalized probabilities.
        return exp_z / sum_exp_z
    # `z = [[1, 2, 3], [1, 2, 3]]` two samples with three classes
    # 1. Subtract the max value along each sample:
    #    z = [[-2, -1, 0], [-2, -1, 0]]
    # 2. exp_z = [[0.13533528, 0.36787944, 1.], [0.13533528, 0.36787944, 1.]]
    # 3. sum_exp_z = [[1.50321472], [1.50321472]]
    # 4. softmax(z) = [[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]]

    def fit(self, X, y):  # data (X, y).
        # TODO Error checks
        y = y.astype(int)  # Convert the labels to integer type in case they are not for indexing
        # Get the dimensions of the input data: number of samples and number of features.
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))  # Get the number of unique labels, which equals the number of classes.
        self.weights = np.zeros((num_features, num_classes))  # Initialize the weights matrix with zeros.
        self.bias = np.zeros(num_classes)  # Initialize the bias vector with zeros.
        # Create a one-hot encoded matrix of labels.
        # For each label, set the corresponding column in the matrix to 1, others to 0.
        # For example, if y =[0, 2, 1] and num_classes = 3, y_one_hot would be:
        # [
        # [1., 0., 0.], `index0`
        # [0., 0., 1.], `index2`
        # [0., 1., 0.]  `index1`
        # ]
        y_one_hot = np.eye(num_classes)[y]
        # Loop over the specified number of iterations to optimize the weights and bias.
        for _ in range(self.num_iterations):
            # Linear combination of inputs and weights, where
            # - X: input data matrix, where each row is a data point and each column is a feature.
            # - self.weights: vector of weights, where each weight corresponds to a feature
            # - np.dot(X, self.weights): This computes the dot product of data matrix and weight vector.
            #   The dot product here essentially computes the weighted sum of input features for each data point.
            # - self.bias: allows model to have some flexibility, it shifts the linear combination by a constant.
            # - linear_model = np.dot(X, self.weights) + self.bias:
            # The final linear combination is the weighted sum of features plus bias term.
            linear_model = np.dot(X, self.weights) + self.bias
            # Applies the softmax function to the linear combination computed above.
            # holds the probabilities of each class for each data point.
            # The sum of probabilities for each data point should be 1
            probabilities = self.softmax(linear_model)
            # Compute the error between the predicted probabilities and the true labels.
            # y_one_hot: True labels, one-hot (search np.eye)
            # By subtracting y_one_hot from probabilities, we are computing the difference
            # between our model's predictions and the true labels
            # backpropagation should happen with this error to update the model's weights and bias terms.
            # This way, with each training iteration, our model learns from its mistakes and adjusts its
            # parameters to minimize the error, and consequently, the loss function.
            error = probabilities - y_one_hot
            # These gradients are used in the subsequent steps to update the model's parameters,
            # moving them in the direction that minimizes the loss function, which is the essence
            # of gradient descent optimization.
            #
            # gradients of the loss function with respect to the model's parameters (weights and bias).
            # These gradients are essential for updating the model's parameters during the backpropagation
            # step in the training process; derivative of the loss function:
            # ∇𝑤(𝐿) = (1 / 𝑛) * np.dot(X.T, error) + 𝜆 * self.weights
            #    `(1 / 𝑛)` normalizes the gradient
            #    - np.dot(X.T, error): This term computes the matrix product of the transpose of the
            #      input data matrix X and the error vector. This is the gradient of the loss with respect
            #      to the weights before regularization.
            #    - self.regularization_strength * self.weights: This term is the regularization part of the
            #      gradient, which helps prevent overfitting by penalizing large weights.
            gradient_weights = (1 / num_samples) * np.dot(X.T, error) + self.regularization_strength * self.weights
            #    This line computes the gradient of the loss function with respect to the bias.
            #      Formula: ∇𝑏(𝐿) = (1/𝑛) * Σ error
            gradient_bias = (1 / num_samples) * np.sum(error, axis=0)
            # Update the weights and bias using the computed gradients and the learning rate.
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    # The predict method is designed to make predictions on the input data X
    # using the trained model parameters (weights and bias).
    def predict(self, X):
        # 1. computing the linear model,
        # 2. applying the softmax function to obtain probabilities,
        # 3. selecting the class with the highest probability.

        # The linear model computes a linear combination of the input features and the model's weights,
        # with the bias term added.
        # Formula: linear_model = 𝑋 • 𝑤 + 𝑏
        # - 𝑋: Input data matrix, where each row is a data point and each column is a feature.
        # - 𝑤: Weight vector of the model.
        # - 𝑏: Bias term of the model.
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(linear_model)
        # The argmax function is used to select the class with the highest probability for each data point.
        # This is the final predicted class for each data point.
        # `axis=1` selects the highest value along each row.
        return np.argmax(probabilities, axis=1)


# This function calculates the precision, recall, and F1 score for a specific class label
# based on the true labels and predicted labels. These metrics are crucial for evaluating
# the performance of classification models, especially in imbalanced datasets.
    # - y_true: The true labels of the data.
    # - y_pred: The labels predicted by the model.
    # - label: The specific class label for which we are calculating the metrics.
def calculate_metrics(y_true, y_pred, label):
    # Step 1: Calculate True Positives (TP)
    # True Positives are the correctly predicted instances of the specified class label.
    # Formula: TP = Σ(y_true == label & y_pred == label)
    tp = np.sum((y_true == label) & (y_pred == label))
    # Step 2: Calculate False Positives (FP)
    # False Positives are the instances incorrectly predicted as the specified class label.
    # Formula: FP = Σ(y_true ≠ label & y_pred == label)
    fp = np.sum((y_true != label) & (y_pred == label))
    # Step 3: Calculate False Negatives (FN)
    # False Negatives are the instances of the specified class label incorrectly predicted as other labels.
    # Formula: FN = Σ(y_true == label & y_pred ≠ label)
    fn = np.sum((y_true == label) & (y_pred != label))
    # Step 4: Calculate Precision
    # Precision is the ratio of True Positives to the sum of True Positives and False Positives.
    # It measures the accuracy of the positive predictions.
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    # Step 5: Calculate Recall
    # Recall is the ratio of True Positives to the sum of True Positives and False Negatives.
    # It measures the ability of the model to identify all relevant instances.
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    # Step 6: Calculate F1 Score
    # The F1 Score is the harmonic mean of precision and recall, which provides a balance between the two metrics.
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


# classification report for a multi-class classification provides key metrics,
# Precision, Recall, and F1 Score for each class label
    # Inputs:
    # - y_true: true labels
    # - y_pred: predicted labels
def classification_report_custom(y_true, y_pred):
    # Identify all the unique class labels present in the true labels dataset.
    labels = np.unique(y_true)

    # Iterate Over Each Unique Label
    for label in labels:
        precision, recall, f1 = calculate_metrics(y_true, y_pred, label)
        # Display Metrics for Each Label
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
            # We already had code to split data in our labs so I used a library
            X_train, X_val, y_train, y_val = train_test_split(
                self.training_set['data'],
                self.training_set['labels'],
                test_size=0.2, random_state=42
            )

            # Adding XGBoost to the model list
            xg_reg = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
                                       max_depth=5, alpha=10, n_estimators=10)

            # Run gridsearch.py before this line
            grid_search_rf = joblib.load('grid_search_rf_final.joblib')
            best_rf = grid_search_rf.best_estimator_

            baseline_models = [
                # Random choices
                ('SimpleDummy', SimpleDummyClassifier()),
                # A custom variant of logistic regression tailored to adapt multiclasses
                ('SoftLogisticRegression', SoftLogisticRegression()),
                # Random choices
                ('Dummy', DummyClassifier(strategy="uniform")),
                # Linear model with stochastic gradient descent learning, with
                # balanced class weights to correct for any imbalance in the dataset.
                ('SGD', SGDClassifier(class_weight='balanced')),
                # Classifier that finds an optimal hyperplane for class separation,
                # applying balanced weights to account for unequal class representation.
                ('SVC', SVC(class_weight='balanced')),
                # Pre-optimized RandomForest model, indicating prior hyperparameter tuning
                ('RandomForest_best_rf', best_rf),
                # An ensemble of decision trees with class weights balanced to improve the
                # decision-making process where data is skewed across different classes.
                ('RandomForest', RandomForestClassifier(class_weight='balanced')),
                # Logistic regression model configured for a high number of iterations and
                # balanced class weights, suitable for complex or imbalanced datasets.
                ('LogisticRegression', LogisticRegression(max_iter=1000, class_weight='balanced')),
                # Gradient boosting framework that is designed for speed and
                # performance, often outperforming other models on structured datasets
                # Not well used, because we tried with Optuna library
                # to get a hyperparameter tuning but random forest tuned was still better
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
