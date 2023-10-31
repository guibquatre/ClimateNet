import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score
import csv
import numpy as np

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
logging.basicConfig(level=logging.INFO)

data_path = "/content/drive/MyDrive/climateDoc/classification-of-extreme-weather-events-udem"
file_names = {'train': 'train.csv', 'test': 'test.csv'}

loader = DataLoader()
train_data, train_labels = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['train']), True)

X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Define an expanded grid of hyperparameters for tuning the RandomForestClassifier.
# This grid will be explored during the grid search to find the best performing set of hyperparameters.
param_grid_rf = {
    # More trees may increase accuracy but also computational cost.
    'n_estimators': [10, 50, 100, 200, 300],  # Number of trees in the forest.
    # Criterion to split on at each node.
    # 'gini' refers to Gini Impurity which is a measure of misclassification,
    # indicating how mixed the classes are in two groups created by a potential split.
    # A Gini Impurity of 0 indicates perfect separation of classes.
    # 'entropy' refers to Information Gain which measures the reduction in entropy (disorder)
    # achieved by partitioning the dataset.
    # A higher information gain indicates a better split that results in purer subgroups.
    # Both 'gini' and 'entropy' are heuristics used to select the best split at each node by
    # evaluating the splits on all features and all possible threshold values for those features.
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of the trees.
    # None means nodes are expanded until they contain less than min_samples_split samples.
    'min_samples_split': [2, 5, 10, 15],  # Minimum number of samples required to split an internal node.
    'min_samples_leaf': [1, 2, 4, 8],  # Minimum number of samples required to be at a leaf node.
    'bootstrap': [True, False],  # Method for sampling data points (with or without replacement).
    'class_weight': [None, 'balanced', 'balanced_subsample'],  # Weights associated with classes
    'max_features': ['auto', 'sqrt', 'log2'],  # The number of features to consider when looking for the best split.
    'max_leaf_nodes': [None, 10, 50, 100],  # Grow trees with a certain maximum number of leaf nodes.
    # Splitting node only if this split induces a decrease of the impurity greater than or equal to this value.
    'min_impurity_decrease': [0.0, 0.01, 0.05]
}

rf = RandomForestClassifier(random_state=42)

f1_scorer = make_scorer(f1_score, average='weighted')

# Using RandomizedSearchCV instead of GridSearchCV
random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf,
                                     n_iter=100,  # You can adjust the number of iterations
                                     scoring=f1_scorer, cv=5, verbose=4, n_jobs=-1, random_state=42)

logging.info("Starting Randomized Search...")

random_search_rf.fit(X_train, y_train)

logging.info("Randomized Search complete.")

joblib.dump(random_search_rf, '/content/drive/MyDrive/climateDoc/saved_models/random_search_rf.joblib')

logging.info("Saved RandomizedSearchCV object to random_search_rf.joblib.")