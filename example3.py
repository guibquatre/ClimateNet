# - os: provides a way of interacting with the operating system, like file operations.
# - logging: enables logging capabilities for debugging and monitoring.
# - csv: provides functionalities to read and write CSV files.
# - yaml: a library for YAML parsing, often used for configuration files.
import os
import logging
import csv
import yaml

# Import project-specific libraries - ClimateDatasetLabeled and ClimateDataset: custom classes for handling labeled
# and unlabeled climate datasets. - CGNet: a custom neural network model for climate analysis. - Config: a utility
# class for loading configuration settings. - track_events, analyze_events, visualize_events: custom functions for
# tracking, analyzing, and visualizing climate events.
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events

# Import machine learning libraries from scikit-learn
# - LogisticRegression: the logistic regression algorithm for classification.
# - DummyClassifier: a simple classifier that can be used for baseline performance.
# - classification_report: a utility for generating performance metrics.
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


# Mount Google Drive
drive.mount('/content/drive')

# Configure the logging system using `logging.basicConfig()`:
#
# - `logging.basicConfig()`: This function sets up the root logger and configures
#    how logging will behave throughout the script.
#
# - `level=logging.INFO`: This sets the minimum logging level to "INFO".
#   The logger will process all log messages with a level of "INFO" or higher.
#   Available levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
#   Messages with a severity lower than "INFO" (like "DEBUG") will be ignored.
#
# - `format='%(asctime)s - %(levelname)s - %(message)s'`: This string sets the
#   format of the log messages.
#   - `%(asctime)s`: Inserts the time when the log message is created.
#   - `%(levelname)s`: Inserts the string representation of the level (e.g., 'INFO', 'ERROR').
#   - `%(message)s`: Inserts the actual log message that is generated in the code.
#
# So when we log a message like `logging.info("Datasets successfully loaded.")`,
# it will appear in the log output as "[Time-Stamp] - INFO - Datasets successfully loaded."
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to download files from URLs
def download_file_from_url(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    !gdown -O {os.path.join(target_dir, url.split('/')[-1])} {url}


# Function to apply YAML config

def apply_yaml_config(yaml_file_path):
    with open(yaml_file_path, "r") as yaml_file:
        env_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    for key, value in env_config.items():
        if key == "env_var":
            os.environ[key] = value


# Function to download and apply config
def download_and_apply_config(url, yaml_file_path, target_dir):
    download_file_from_url(url, target_dir)
    apply_yaml_config(yaml_file_path)


def _save_submission(predictions):
    """Private method to save predictions."""
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SNo', 'Label'])
        for idx, label in enumerate(predictions, 1):
            writer.writerow([idx, label])


class ClimateAnalysisPipeline:
    def __init__(self, config_path):
        self.inference_set = None
        self.training_set = None
        try:
            self.config = Config(config_path)
            self.model = CGNet(self.config)
        except Exception as e:
            logging.error(f"Initialization error: {e}")

    def load_datasets(self, training_path, inference_path):
        """Load datasets."""
        try:
            self.training_set = ClimateDatasetLabeled(training_path, self.model.config)
            self.inference_set = ClimateDataset(inference_path, self.model.config)
            logging.info("Datasets successfully loaded.")
        except Exception as e:
            logging.error(f"Dataset loading error: {e}")

    def train_and_evaluate(self):
        """
        Train the main model and evaluate it against baseline models.
        This is a part of the ClimateAnalysisPipeline class.
        """

        # Train the primary climate model. - `self.model` refers to the CGNet model that was initialized in the
        # __init__ method. - `self.training_set` is the dataset used for training, which should be loaded beforehand
        # using the `load_datasets()` method. - The `.train()` method trains the model using the provided training set.
        self.model.train(self.training_set)

        # Evaluate the model against a Dummy Classifier. - A Dummy Classifier is a simple classifier that provides a
        # baseline to compare with the main model. - The strategy used here is "uniform", meaning it generates
        # predictions uniformly at random. - The `_evaluate_baseline()` method evaluates the performance of the given
        # model on the inference dataset and logs it.
        self._evaluate_baseline(DummyClassifier(strategy="uniform"), "Dummy Classifier")

        # Evaluate the model against Logistic Regression. - Logistic Regression is a more advanced baseline model. -
        # `max_iter=1000` specifies the maximum number of iterations for the logistic regression solver. - The
        # `_evaluate_baseline()` method evaluates the performance of the given model on the inference dataset and
        # logs it.
        self._evaluate_baseline(LogisticRegression(max_iter=1000), "Logistic Regression")

    def _evaluate_baseline(self, model, model_name):
        """
        Private method for evaluating baseline models.
        This is a part of the ClimateAnalysisPipeline class.
        """

        # Fit the model to the training data.
        # - The `model` parameter is the machine learning model we want to evaluate.
        # - `self.training_set.data` contains the features of the training dataset.
        # - `self.training_set.labels` contains the labels of the training dataset.
        # - The `fit()` method trains the model using the provided training set features and labels.
        model.fit(self.training_set.data, self.training_set.labels)

        # Make predictions on the inference set.
        # - `self.inference_set.data` contains the features of the dataset that we want to make predictions on.
        # - The `predict()` method uses the trained model to make predictions.
        predictions = model.predict(self.inference_set.data)

        # Log the performance of the model. - `model_name` is a string representing the name of the model being
        # evaluated, e.g., "Dummy Classifier". - `logging.info()` logs the message at the INFO level. - This log
        # entry serves as a heading in the logs to indicate which model's performance is being displayed next.
        logging.info(f"{model_name} Performance:")

        # Log the detailed classification report. - `classification_report` is a utility function from scikit-learn
        # that generates a text report showing the classification metrics. - `self.inference_set.labels` contains the
        # true labels of the inference set for comparison. - The report shows key metrics like precision, recall,
        # and F1-score, providing insights into the model's performance.
        logging.info(classification_report(self.inference_set.labels, predictions))

    def predict_and_analyze(self):
        """
        Predict, analyze, and visualize results.
        This is part of the ClimateAnalysisPipeline class.
        """

        # Make predictions on the inference set.
        # - `self.model` is our trained machine learning model.
        # - `self.inference_set` contains the data for which we want to make predictions.
        # - The `predict()` method will use the trained model to make predictions on this set.
        predictions = self.model.predict(self.inference_set)

        # Generate event masks based on the predictions.
        # - `track_events()` is a custom function that takes predictions as input.
        # - It likely returns a list/array of 'masks' that indicate which events are happening at each time step.
        event_masks = track_events(predictions)

        # Analyze the events based on the generated event masks and predictions. - `analyze_events()` is a custom
        # function. - This probably analyzes the frequency, distribution, or some other statistical properties of the
        # tracked events.
        analyze_events(event_masks, predictions)

        # Visualize the tracked events.
        # - `visualize_events()` is a custom function that likely produces plots or other visual representations.
        # - This function uses the event masks and the inference set to create these visualizations.
        visualize_events(event_masks, self.inference_set)

        # Save the predictions to a CSV file.
        # - `_save_submission()` is a private function that writes the predictions into a CSV file.
        # - This is likely for submission to a competition or for record-keeping.
        _save_submission(predictions)

        # Log that the predictions were successfully generated, analyzed, and visualized.
        # - This log message serves as a confirmation that the entire pipeline of operations was completed successfully.
        logging.info("Predictions successfully generated, analyzed, and visualized.")

    def save_model(self, path):
        """Save model."""
        self.model.save_model(path)
        logging.info("Model saved at: {path}")


def main():
    """
    Main function to run the pipeline.
    This function serves as the entry point for the entire Climate Analysis Pipeline.
    """

    try:
        # Define constants for configuration and dataset paths.
        # - CONFIG_PATH holds the path to the JSON file containing various configurations for the pipeline.
        # - TRAINING_SET_PATH and INFERENCE_SET_PATH specify where the training and test datasets are located.
        # - SAVE_MODEL_PATH is where we will save the trained machine learning model.
        CONFIG_PATH = './config.json'
        TRAINING_SET_PATH = './classification-of-extreme-weather-events-udem/train.csv'
        INFERENCE_SET_PATH = './classification-of-extreme-weather-events-udem/test.csv'
        SAVE_MODEL_PATH = './model'

        # Initialize the Climate Analysis Pipeline with configurations.
        # - Reads configurations from the file specified by CONFIG_PATH.
        pipeline = ClimateAnalysisPipeline(CONFIG_PATH)

        # Load datasets for training and inference.
        # - Reads the training and inference datasets from the CSV files specified by the paths.
        pipeline.load_datasets(TRAINING_SET_PATH, INFERENCE_SET_PATH)

        # Train the model and evaluate it against baseline models.
        # - Trains the model on the training set and evaluates its performance.
        pipeline.train_and_evaluate()

        # Make predictions, analyze, and visualize the results.
        # - Uses the trained model to make predictions on the inference set and analyzes the results.
        pipeline.predict_and_analyze()

        # Save the trained model.
        # - Saves the trained model to the path specified by SAVE_MODEL_PATH for future use or deployment.
        pipeline.save_model(SAVE_MODEL_PATH)

        # Log that the entire pipeline has been successfully executed.
        # - This serves as a confirmation that all the steps were completed without issues.
        logging.info("Climate Analysis Pipeline execution complete.")

    except RuntimeError as e:
        # Handle any RuntimeErrors that may occur during the execution of the pipeline.
        # - Prints the error message to the console and logs it as an error for further investigation.
        print(e)
        logging.error(f"Main execution error: {e}")


if __name__ == '__main__':
    main()

