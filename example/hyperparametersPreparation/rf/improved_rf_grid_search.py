import joblib
import os
import logging
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score

save_dir = "../../saved_models/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fit_counter = [0]  # Using a list to hold the counter so it can be modified inside the function
best_score = [0]  # Using a list to store the best score

def save_best_model(clf, score):
    filename = f"{save_dir}model_{fit_counter[0]}_score_{score:.4f}.joblib"
    joblib.dump(clf, filename)
    print(f"Model saved: {filename}")


def custom_scorer(clf, X, y_true):
    y_pred = clf.predict(X)
    score = f1_score(y_true, y_pred, average='weighted')

    fit_counter[0] += 1

    if fit_counter[0] % 10000 == 0:
        if score > best_score[0]:
            save_best_model(clf, score)
            best_score[0] = score

    if fit_counter[0] % 1000 == 0:
        print("\n" + "*" * 30)
        print(f"Processed {fit_counter[0]} fits, Best Score: {best_score[0]:.4f}")
        print("*" * 30 + "\n")

    return score


logging.basicConfig(level=logging.INFO)

class DataLoader:
    @staticmethod
    def safe_load_csv_dataset(file_path: str, is_train=True):
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)
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


data_path = "../../classification-of-extreme-weather-events-udem"
file_names = {'train': 'train.csv', 'test': 'test.csv'}

loader = DataLoader()

train_data, train_labels = loader.safe_load_csv_dataset(os.path.join(data_path, file_names['train']), True)

X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42
)

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
f1_scorer = make_scorer(custom_scorer, greater_is_better=True)

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf,
                              scoring=f1_scorer,
                              cv=5,
                              verbose=5,
                              n_jobs=-1)

logging.info("Starting Grid Search...")

grid_search_rf.fit(X_train, y_train)

logging.info("Grid Search complete.")

# Saving the best model from all fits
joblib.dump(grid_search_rf.best_estimator_, os.path.join(save_dir, 'best_of_all_fits_model.joblib'))

logging.info("Saved the best of all fits model.")
