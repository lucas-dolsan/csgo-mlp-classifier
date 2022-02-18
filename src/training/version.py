import os
from sklearn.neural_network import MLPClassifier
from pathlib import Path
from misc import serializer
import config
import json

class Version:
  def __init__(
    self,
    version_name: str,
    classifier_instance: MLPClassifier,
    used_parameters: dict,
    used_dataset: str,
    metrics: dict
  ):
    self.version_name = version_name
    self.classifier_instance = classifier_instance
    self.used_parameters = used_parameters
    self.used_dataset = used_dataset
    self.metrics = metrics

  def save(self):
    full_version_name = f'{self.version_name}-{self.used_dataset.replace(".csv", "")}'
    
    if config.VERBOSE:
      print(f'saving version... ({full_version_name})')

    folder_version=f'versions/{full_version_name}'

    if os.path.isdir(folder_version):
      raise Exception(f'Version {full_version_name} already exists')

    # cria o folder da version
    Path(folder_version).mkdir(parents=True, exist_ok=True)

    # grava os parametros usados em um arquivo json
    with open(f'{folder_version}/parameters.json', 'w') as fp:
      json.dump(self.used_parameters, fp)

    # grava as métricas geradas em um arquivo json
    with open(f'{folder_version}/metrics.json', 'w') as fp:
      json.dump(self.metrics, fp)

    # serializa o modelo treinado do classificador
    serializer.serialize(
      f'{folder_version}/classifier_model_{self.version_name}',
      self.classifier_instance
    )