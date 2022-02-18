from dataset.dataset_handler import DatasetHandler
from training.trainer import Trainer
from training.training_parameters import parameters
from evaluation.metrics import build_metrics
from training.version import Version
from training.training_plan import training_plan
import config

def train_classifier(version_name, parameters):
  dataset = DatasetHandler.read_dataset()
  x, y = DatasetHandler.preprocess_dataset(dataset)
  
  x_train, x_test, y_train, y_test = DatasetHandler.organize_dataset(x,y)

  trained_mlp_classifier = Trainer.run_training(parameters, x_train, y_train)

  metrics = build_metrics(
    x_test,
    y_test,
    trained_mlp_classifier.predict(x_test),
    trained_mlp_classifier
  )

  version = Version(
    version_name=version_name,
    classifier_instance=trained_mlp_classifier,
    used_parameters=parameters,
    used_dataset=config.DATASET_FILE,
    metrics=metrics
  )

  version.save()


for version_name, parameters in enumerate(training_plan):
  train_classifier(version_name, parameters)