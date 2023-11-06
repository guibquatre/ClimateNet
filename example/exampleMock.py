# Placeholder Imports already imported


class DataLoader:
    @staticmethod
    def safe_load_csv_dataset(file_path: str, is_train=True):
        # Placeholder Already Implemented
        pass


class SubmissionSaver:
    @staticmethod
    def save_submission(predictions, file_path='submission.csv'):
        # Placeholder Already Implemented
        pass


class ModelSaver:
    @staticmethod
    def save_model(_model):
        # Placeholder Already Implemented
        pass


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
        return (X - self.mean) / (self.std + 1e-8)

    def normalize(self, X):
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
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_strength=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None

    @staticmethod
    def softmax(z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        return exp_z / sum_exp_z

    def fit(self, X, y):
        y = y.astype(int)
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)
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
    for label in labels:
        precision, recall, f1 = calculate_metrics(y_true, y_pred, label)
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

            # xg_reg = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
            #                            max_depth=5, alpha=10, n_estimators=10)
            grid_search_rf = joblib.load('saved_models/grid_search_rf_final.joblib')
            best_rf = grid_search_rf.best_estimator_

            baseline_models = [
                # ('SimpleDummy', SimpleDummyClassifier()),
                # ('SoftLogisticRegression', SoftLogisticRegression()),
                # ('Dummy', DummyClassifier(strategy="uniform")),
                # ('SGD', SGDClassifier(class_weight='balanced')),
                # ('SVC', SVC(class_weight='balanced')),
                ('RandomForest_best_rf', best_rf),
                ('RandomForest', RandomForestClassifier(class_weight='balanced'))
                # ('LogisticRegression', LogisticRegression(max_iter=1000, class_weight='balanced')),
                # ('XGBoost', xg_reg)
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
    preprocessor.fit(train_data)
    train_data = preprocessor.transform(train_data, method='standardize')
    test_data = preprocessor.transform(test_data, method='standardize')

    pipeline = ClimateAnalysisPipeline()

    pipeline.training_set['data'], pipeline.training_set['labels'] = train_data, train_labels
    pipeline.inference_set['data'] = test_data

    best_model = pipeline.train_and_evaluate()

    if best_model:
        test_predictions = best_model.predict(test_data)
        SubmissionSaver.save_submission(test_predictions)
    else:
        logging.error("No best model, sorry")


if __name__ == '__main__':
    main()
