import joblib
import os
import logging
import xgboost as xgb
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

logging.basicConfig(level=logging.INFO)

data_path = "/content/drive/MyDrive/climateDoc/classification-of-extreme-weather-events-udem"
file_names = {'train': 'train.csv', 'test': 'test.csv'}

loader = DataLoader()
train_data, train_labels = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['train']), True)

X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
param_grid_xgb = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 12],
    'colsample_bytree': [0.3, 0.5, 0.7, 0.9, 1],
    'subsample': [0.3, 0.5, 0.7, 0.9, 1],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5, 1, 2],
    'min_child_weight': [1, 2, 3, 4, 5, 6],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],  # L1 regularization term on weights
    'reg_lambda': [1, 1.5, 2, 3, 4.5]  # L2 regularization term on weights
}

xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

f1_scorer = make_scorer(f1_score, average='weighted')

# Using RandomizedSearchCV with the XGBoost classifier
random_search_xgb = RandomizedSearchCV(estimator=xgb_clf, param_distributions=param_grid_xgb,
                                      n_iter=200,  # increased iterations
                                      scoring=f1_scorer, cv=5, verbose=4, n_jobs=-1, random_state=42)

logging.info("Starting Randomized Search for XGBoost...")

random_search_xgb.fit(X_train, y_train)

logging.info("Randomized Search for XGBoost complete.")

joblib.dump(random_search_xgb, '/content/drive/MyDrive/climateDoc/saved_models/random_search_xgb.joblib')

logging.info("Saved RandomizedSearchCV object for XGBoost to random_search_xgb.joblib.")
