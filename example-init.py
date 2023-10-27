from competition.ClimateNet.example.climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from competition.ClimateNet.example.climatenet.models import CGNet
from competition.ClimateNet.example.climatenet.utils.utils import Config
from competition.ClimateNet.example.climatenet import track_events
from competition.ClimateNet.example.climatenet import analyze_events
from competition.ClimateNet.example.climatenet import visualize_events

from os import path
# may you please give me an improve submittable python file from the requirements and this code as example? do you best thank you
config = Config('example/model/config-init.json')
cgnet = CGNet(config)

train_path = 'PATH_TO_TRAINING_SET'
inference_path = 'PATH_TO_INFERENCE_SET'

train = ClimateDatasetLabeled(path.join(train_path, 'train'), config)
test = ClimateDatasetLabeled(path.join(train_path, 'test'), config)
inference = ClimateDataset(inference_path, config)

cgnet.train(train)
cgnet.evaluate(test)

cgnet.save_model('trained_cgnet')
# use a saved model with
# cgnet.load_model('trained_cgnet')

class_masks = cgnet.predict(inference) # masks with 1==TC, 2==AR
event_masks = track_events(class_masks) # masks with event IDs

analyze_events(event_masks, class_masks, 'results/')
visualize_events(event_masks, inference, 'pngs/')
