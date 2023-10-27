from os import path
import csv
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


class ClimateAnalysisPipeline:
    def __init__(self, _config_path):
        self.config = Config(_config_path)
        self.model = CGNet(self.config)
        self.training_set = None
        self.inference_set = None

    def load_nc_datasets(self, _training_nc_path, _inference_nc_path):
        try:
            self.training_set = ClimateDatasetLabeled(_training_nc_path, self.config)
            self.inference_set = ClimateDataset(_inference_nc_path, self.config)
            print("Datasets successfully loaded from nc files")
        except Exception as e:
            print("Dataset loading error: %s" % e)

    def train_and_evaluate(self):
        self.model.train(self.training_set)
        self.model.evaluate(self.inference_set)
        # Additional models
        self.evaluate_baseline_models()

    def evaluate_baseline_models(self):
        baseline_models = [
            (DummyClassifier(strategy="uniform"), "Dummy Classifier"),
            (SGDClassifier(), "Stochastic Gradient Descent"),
            (SVC(), "Support Vector Machine"),
            (RandomForestClassifier(), "Random Forest"),
            (LogisticRegression(max_iter=1000), "Logistic Regression"),
        ]

        for model, name in baseline_models:
            model.fit(self.training_set.data, self.training_set.labels)
            _predictions = model.predict(self.inference_set.data)
            print("%s Performance" % name)
            print(classification_report(self.inference_set.labels, _predictions))

    def analyze_and_visualize(self):
        class_masks = self.model.predict(self.inference_set)
        event_masks = track_events(class_masks)
        analyze_events(event_masks, class_masks, 'results/')
        visualize_events(event_masks, self.inference_set, 'pngs/')

    def save_model(self, save_path):
        self.model.save_model(save_path)

    @staticmethod
    def save_submission(_predictions, file_path='submission.csv'):
        with open(file_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SNo', 'Label'])
            for idx, label in enumerate(_predictions, 1):
                writer.writerow([idx, label])


if __name__ == '__main__':
    config_path = '/content/drive/MyDrive/climateDoc/model/config-init.json'
    training_nc_path = '/content/drive/MyDrive/climateDoc/classification-of-extreme-weather-events-udem/train'
    inference_nc_path = '/content/drive/MyDrive/climateDoc/classification-of-extreme-weather-events-udem/test'

    pipeline = ClimateAnalysisPipeline(config_path)
    pipeline.load_nc_datasets(training_nc_path, inference_nc_path)
    pipeline.train_and_evaluate()
    pipeline.analyze_and_visualize()
    pipeline.save_model('trained_cgnet')

    predictions = pipeline.model.predict(pipeline.inference_set)  # Replace with your prediction method
    pipeline.save_submission(predictions, 'submission.csv')
