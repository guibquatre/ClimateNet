import joblib  # saving and loading Python objects efficiently.
import os  # providing a way of using operating system dependent functionality.
from sklearn.ensemble import RandomForestClassifier  # Import the RandomForest algorithm.
from sklearn.model_selection import GridSearchCV, train_test_split  # tools for hyperparameter tuning and splitting
from sklearn.metrics import make_scorer, f1_score  # tools for custom scoring function and the F1 score metric.
import logging  # Import logging to provide event logging to sys.stderr.
from example_combined import DataLoader  # custom data loading class.

# Configure logging to log informational messages.
logging.basicConfig(level=logging.INFO)

# Define the path to the data.
data_path = "/Users/spokboud/INF6390/competition/ClimateNet/example/classification-of-extreme-weather-events-udem"
file_names = {'train': 'train.csv', 'test': 'test.csv'}

# Instantiate a DataLoader object.
loader = DataLoader()

# Use os.path.join to construct a pathname with the path and filename.
# Load the training data and labels using the custom DataLoader class.
train_data, train_labels = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['train']), True)

# Split the training data into training and validation subsets.
# 80% of the data is used for training and 20% is used for validation.
# random_state is a seed value to ensure reproducibility between runs.
X_train, X_val, y_train, y_val = train_test_split(
    train_data,
    train_labels,
    test_size=0.2,
    random_state=42
)

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

# Instantiate a RandomForestClassifier object with a fixed random state for reproducibility.
rf = RandomForestClassifier(random_state=42)

# Create a custom scoring function using the make_scorer function and the F1 score metric.
# The F1 score is a measure model's precision and recall.
f1_scorer = make_scorer(f1_score, average='weighted')

# Instantiate a GridSearchCV object to perform a grid search of the RandomForestClassifier hyperparameters.
# This object will explore the parameter grid using cross-validation to find the best set of hyperparameters.
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf,
                              scoring=f1_scorer,  # Use the custom F1 scoring function.
                              cv=5,  # Perform 5-fold cross-validation.
                              verbose=4,  # Output messages to the console.
                              n_jobs=-1)  # Use all available cores on the machine for parallel processing.

# Log the start of the grid search process to the console.
logging.info("Starting Grid Search...")

# Fit the GridSearchCV object to the training data.
# train a RandomForestClassifier for each combination of hyperparameters in the grid,
# and evaluate them using cross-validation.
grid_search_rf.fit(X_train, y_train)

# Log the completion of the grid search process to the console.
logging.info("Grid Search complete.")

# Save the fitted GridSearchCV object to disk for later use.
# This object contains the best set of hyperparameters found during the grid search,
# as well as the fitted RandomForestClassifier with those hyperparameters.
joblib.dump(grid_search_rf, 'model/grid_search_rf.joblib')

# Log the saving of the GridSearchCV object to the console.
logging.info("Saved GridSearchCV object to grid_search_rf.joblib.")
