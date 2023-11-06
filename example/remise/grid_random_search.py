# Look at grid_search.py for comments
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score
from example import DataLoader

logging.basicConfig(level=logging.INFO)

data_path = "classification-of-extreme-weather-events-udem"
file_names = {'train': 'train.csv', 'test': 'test.csv'}

loader = DataLoader()
train_data, train_labels = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['train']), True)
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

param_grid_rf = {
    'n_estimators': [10, 50, 100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 50, 100],
    'min_impurity_decrease': [0.0, 0.01, 0.05]
}

rf = RandomForestClassifier(random_state=42)

f1_scorer = make_scorer(f1_score, average='weighted')

# Using RandomizedSearchCV instead of GridSearchCV
random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf,
                                     n_iter=100,  # Ajustable
                                     scoring=f1_scorer, cv=5, verbose=4, n_jobs=-1, random_state=42)

logging.info("Starting Randomized Search...")

random_search_rf.fit(X_train, y_train)

logging.info("Randomized Search complete.")

joblib.dump(random_search_rf, 'random_search_rf.joblib')

logging.info("Saved RandomizedSearchCV object to random_search_rf.joblib.")