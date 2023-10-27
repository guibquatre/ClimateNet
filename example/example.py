import os
import csv
import pandas as pd
import logging
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# Initialize logging
logging.basicConfig(level=logging.INFO)


def safe_load_csv_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        labels = data['Label'] if 'Label' in data.columns else None
        features = data.drop(['Label'], axis=1) if labels is not None else data
        return features, labels
    except Exception as e:
        logging.error("Error loading dataset:: %s", str(e))
        return None, None


class ClimateAnalysisPipeline:
    def __init__(self):
        logging.info("Initializing ClimateAnalysisPipeline")
        self.training_set = {'data': None, 'labels': None}
        self.inference_set = {'data': None, 'labels': None}

    def load_datasets(self, _path, _file_names):
        logging.info("Loading datasets from path: %s", _path)
        try:
            self.training_set['data'], self.training_set['labels'] = \
                safe_load_csv_dataset(os.path.join(_path, _file_names['train']))

            inference_data, _ = safe_load_csv_dataset(os.path.join(_path, _file_names['test']))
            if inference_data is not None:
                self.inference_set['data'] = inference_data
            else:
                logging.error("Error in loading inference dataset.")
        except Exception as e:
            logging.error("Error in loading datasets: %s", str(e))
        logging.info("Datasets loaded")

    def train_model(self, model):
        model.fit(self.training_set['data'], self.training_set['labels'])
        return model

    def evaluate_model(self, model):
        _predictions = model.predict(self.inference_set['data'])
        report = classification_report(self.inference_set['labels'], _predictions, output_dict=True)
        return report['weighted avg']['f1-score']

    def train_and_evaluate(self):
        logging.info("Starting model training and evaluation")
        if self.training_set['data'] is not None:
            X_train, X_val, y_train, y_val = train_test_split(self.training_set['data'],
                                                              self.training_set['labels'],
                                                              test_size=0.2, random_state=42)

            baseline_models = [
                ('Dummy', DummyClassifier(strategy="uniform")),
                ('SGD', SGDClassifier()),
                ('SVC', SVC()),
                ('RandomForest', RandomForestClassifier()),
                ('LogisticRegression', LogisticRegression(max_iter=1000))
            ]

            best_f1 = 0.0
            best_model_name = ""
            _best_model = None

            for name, model in baseline_models:
                model.fit(X_train, y_train)
                _predictions = model.predict(X_val)
                report = classification_report(y_val, _predictions, output_dict=True)
                f1 = report['weighted avg']['f1-score']
                logging.info("Model {} F1-score: {:.4f}".format(name, f1))

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = name
                    _best_model = model

            logging.info("Best model is {} with F1-score of {:.4f}".format(best_model_name, best_f1))

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_directory = "./saved_models/"
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

            best_model_filename = "{}_{}_best_model.pkl".format(best_model_name, timestamp)
            best_model_path = os.path.join(model_directory, best_model_filename)

            with open(best_model_path, 'wb') as f:
                pickle.dump(_best_model, f)

            logging.info("Best model saved as {}".format(best_model_filename))

            return _best_model
        else:
            logging.error("Training data is missing. Cannot proceed.")
            return None

    @staticmethod
    def save_submission(_predictions):
        logging.info("Saving submission")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        submission_file_path = "./submission_{}.csv".format(timestamp)
        try:
            with open(submission_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['SNo', 'Label'])
                for idx, label in enumerate(_predictions, 1):
                    writer.writerow([idx, label])
            logging.info("Submission saved as {}".format(submission_file_path))
        except Exception as e:
            logging.error("Error saving the submission: %s", str(e))


if __name__ == '__main__':
    logging.info("Starting main execution")
    data_path = "/home/gui/INF6390/competition/ClimateNet/example/classification-of-extreme-weather-events-udem/"
    # data_path = '/content/drive/MyDrive/climateDoc/classification-of-extreme-weather-events-udem/'
    file_names = {'train': 'train.csv', 'test': 'test.csv'}

    pipeline = ClimateAnalysisPipeline()
    pipeline.load_datasets(data_path, file_names)
    best_model = pipeline.train_and_evaluate()
    if best_model:
        predictions = best_model.predict(pipeline.inference_set['data'])
        logging.info("Predictions made")
        pipeline.save_submission(predictions)
    logging.info("Ending pipeline!")
