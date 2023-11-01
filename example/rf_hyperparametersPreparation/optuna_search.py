import joblib
import os
import logging
import csv
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, cross_val_score


class DataLoader:
    @staticmethod
    def safe_load_csv_dataset(file_path: str, is_train=True):
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                data = np.array([row for row in reader], dtype=float)

            if is_train:
                labels = data[:, -1]
                features = data[:, :-1]
            else:
                labels = None
                features = data

            return features, labels

        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            return None, None
        except ValueError as e:
            logging.error(f"Value error: {str(e)}")
            return None, None


# Objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 300),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 5, 40, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5, 100, log=True),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.05)
    }

    clf = RandomForestClassifier(**params, random_state=42)
    return cross_val_score(clf, X_train, y_train, scoring=f1_scorer, cv=5, n_jobs=-1).mean()


logging.basicConfig(level=logging.INFO)

data_path = "/content/drive/MyDrive/climateDoc/classification-of-extreme-weather-events-udem"
file_names = {'train': 'train.csv', 'test': 'test.csv'}

loader = DataLoader()
train_data, train_labels = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['train']), True)
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')

logging.info("Starting Optuna optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

logging.info("Optuna optimization complete.")

best_rf = RandomForestClassifier(**study.best_params, random_state=42)
best_rf.fit(X_train, y_train)

joblib.dump(best_rf, '/content/drive/MyDrive/climateDoc/saved_models/best_rf_optuna.joblib')
logging.info("Saved RandomForest model with Optuna-optimized hyperparameters to best_rf_optuna.joblib.")
