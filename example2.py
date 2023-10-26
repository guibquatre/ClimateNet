# We start by importing various tools and classes that will help us throughout our code.
# These are specialized functions and classes built for handling climate data, models, and configurations.

# These are for handling the climate dataset.
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset

# This is the actual machine learning model we'll be using, called CGNet.
from climatenet.models import CGNet

# Utilities for loading configurations.
from climatenet.utils.utils import Config

# These are specialized functions for tracking, analyzing, and visualizing weather events.
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events

# For file path manipulation
from os import path


# We load settings from a JSON file into our 'config' object.
# This file likely contains settings like learning rate, batch size, etc., for our machine learning model.
config = Config('config.json')


# Create an instance of CGNet (our machine learning model) using the settings we loaded.
cgnet = CGNet(config)


# Define where our training data is located.
# We'll use this data to 'teach' our model.
train_path = './train/'


# Define where our inference data is located.
# Inference data is what we'll use to make predictions after our model has learned from the training data.
inference_path = 'PATH_TO_INFERENCE_SET'


# We create labeled datasets for training and testing.
# This data has both the inputs and the correct answers (labels), so our model can learn effectively.
train = ClimateDatasetLabeled(path.join(train_path, 'train'), config)
test = ClimateDatasetLabeled(path.join(train_path, 'test'), config)


# Create a dataset without labels for inference.
# This is the data we'll feed into our trained model to make predictions.
inference = ClimateDataset(inference_path, config)


# Now we actually train our CGNet model.
# The model will learn from the training data we provided.
cgnet.train(train)


# After training, we test the model on new data to see how well it has learned.
# This helps us understand if our model is good or needs improvement.
cgnet.evaluate(test)


# Save our trained model to disk.
# This allows us to use this same trained model later without having to retrain it.
cgnet.save_model('trained_cgnet')


# If we already have a trained model saved, we can load it with this line.
# cgnet.load_model('trained_cgnet')


# Use our trained model to make predictions on new, unlabeled data.
# 'class_masks' will contain the predictions. For instance, 1 for Tropical Cyclones, 2 for Atmospheric Rivers.
class_masks = cgnet.predict(inference)


# We identify unique events in our predictions.
# This can help us understand patterns or sequences in the weather events we've predicted.
event_masks = track_events(class_masks)


# We analyze the tracked events and save the analysis results.
# This could be metrics or characteristics about the weather events we've identified.
analyze_events(event_masks, class_masks, 'results/')


# Finally, we visualize our findings.
# This could create charts, graphs, or other visual representations to better understand our results.
visualize_events(event_masks, inference, 'pngs/')
