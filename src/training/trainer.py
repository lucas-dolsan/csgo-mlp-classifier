from sklearn.neural_network import MLPClassifier
import json
import config

def _print_parameters(parameters: dict):
    print(json.dumps(parameters, indent=2).replace('"', ''))
    
class Trainer:

  # essa função vai retornar um classifier treinado de acordo com os parâmetros especificados
  def run_training(parameters, x_train=None, y_train=None) -> MLPClassifier:

      if config.VERBOSE:
        print(f'training... ({config.VERSION_NAME})')
        print(f'using parameters: ')
        _print_parameters(parameters)

      mlp_classifier = MLPClassifier(**parameters).fit(x_train, y_train)

      return mlp_classifier
