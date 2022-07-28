from dataset.dataset_handler import DatasetHandler
from training.trainer import Trainer
from training.training_parameters import parameters
from evaluation.evaluation import evaluate_classifier 
from training.version import Version
from training.training_plan import training_plan
import config

def train_classifier(version_name, parameters):
  dataset = DatasetHandler.read_dataset()
  x, y = DatasetHandler.preprocess_dataset(dataset)
  
  x_train, x_test, y_train, y_test = DatasetHandler.organize_dataset(x,y)

  classifier = Trainer.run_training(version_name, parameters, x_train, y_train)

  metrics = evaluate_classifier(x_test, y_test, classifier)

  version = Version(
    version_name=version_name,
    classifier_instance=classifier,
    used_parameters=parameters,
    used_dataset=config.DATASET_FILE,
    metrics=metrics
  )

  version.save()

# Maior numero de versão existente 
latest_version=8

for version, parameters in enumerate(training_plan):
  train_classifier(f'v{version + latest_version + 1}', parameters)