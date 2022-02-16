
# arg: test_size: define a porcentagem dos dados totais do dataset que serão usados para teste
# ex.: com test_size: 0.2: 80% do dataset será usado para treinamento (x_train), e o restante (20%)
# será reservado para ser usado no teste (x_test)
#
# x_train (dados para treino)
# y_train (objetivos para treino)
# x_test (dados para teste)
# y_test (objetivos para teste)
TEST_SIZE = 0.33
# habilita logs do progresso do script
VERBOSE=True
# Especifica o nome de um arquivo de classifier pré-treinado, se o arquivo não existir, o script irá 
# treinar um novo usando os parâmetros definidos, e salvará num arquivo com o nome especificado.
#Contido dentro da pasta trained-classifiers
CLASSIFIER_FILE = 'v49.pkl'
#nome do dataset a ser utilizado, contido dentro da pasta de datasets
DATASET_FILE = 'small_dataset.csv'
# Se vai embaralhar ou não os dados lidos apartir do CSV
SHUFFLE=True
# Se deve ou não mostrar a matriz de confusão
# numa janela do matplotlib (independente do caso o resultado sairá , stdout do script)
SHOW_MATPLOTLIB_OUTPUT=False
# Se deve ou não normalizar os dados gerados da matriz de confusão
NORMALIZE_CONFUSION_MATRIX=False

# definido de forma arbitrária, para fins do classificador binário
POSITIVE_LABEL='CT'
NEGATIVE_LABEL='TR'

# são as duas classes classificadas (os dois times) 
LABELS=(POSITIVE_LABEL, NEGATIVE_LABEL)

# Misc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

# Preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler

# MLP Classifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from sklearn.neural_network import MLPClassifier

# TODO review
from sklearn.model_selection import train_test_split, cross_val_score

# Essa função lê um classifier serializado pré-treinado
def read_saved_classifier():
    filepath = f'trained-classifiers/{CLASSIFIER_FILE}'

    file_exists = os.path.exists(filepath)

    if not file_exists:
        return None

    with open(filepath, 'rb') as fid:
        return pickle.load(fid)

# Essa função serializa um classifier treinado
def save_classifier(classifier):
    filepath = f'trained-classifiers/{CLASSIFIER_FILE}'

    with open(filepath, 'wb') as fid:
        pickle.dump(classifier, fid)

# Lê o dataset em formato csv, com delimiter=, na pasta de datasets do projeto
def read_dataset():
    if VERBOSE:
      print(f'reading dataset... ({DATASET_FILE})')
    filepath = f'datasets/{DATASET_FILE}'
    return pd.read_csv(filepath, delimiter=',')

# Pré-processa o dataset, preparando os dados para treinamento/testes
def preprocess_dataset(dataset):
    if VERBOSE:
      print('preprocessing dataset...')
    # Precisamos usar o LabelEncoder para mapear a coluna 'map' para valores numéricos
    # ex.: {'de_dust2' -> 01; 'de_mirage' -> 02; etc}

    label_encoder = LabelEncoder()
    label_encoder.fit(dataset['map'])
    dataset['map'] = label_encoder.transform(dataset['map'])

    # para definir x, precisamos  ler todas as colunas, menos a última
    # pois a última coluna ('round_winner') armazena o resultado de cada rodada
    x = dataset[dataset.columns[:-1]]

    # TODO review
    x = StandardScaler().fit_transform(x)

    # resultado de qual time (CT ou TR) venceu a rodada
    y = dataset['round_winner']

    return x, y


# essa função vai retornar um classifier treinado de acordo com os parâmetros especificados
def run_training(x_train, y_train):
    max_iter = x_train.shape[0]
    # max_iter = 500
    batch_size = 'auto'
    hidden_layer_sizes=()

    if VERBOSE:
      print(f'training... ({CLASSIFIER_FILE})')
      print(f'using parameters: ')
      print(f'max_iter:{max_iter:}')
      print(f'batch_size:{batch_size:}')
      print(f'hidden_layer_sizes:{hidden_layer_sizes:}')

    mlp_classifier = MLPClassifier(
      max_iter=max_iter,
      batch_size=batch_size,
      hidden_layer_sizes=hidden_layer_sizes,
      verbose=False
    ).fit(x_train, y_train)

    return mlp_classifier

# Essa função constroi todas as métricas que serão usadas na avaliação do resultado do MLPClassifier
def build_metrics(x_test, y_test, y_pred, classifier):
    if VERBOSE:
      print('building metrics...')

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average='binary', pos_label=POSITIVE_LABEL)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=POSITIVE_LABEL)

    confusion_matrix_args = confusion_matrix(y_test, y_pred).ravel()

    tn, fp, fn, tp = confusion_matrix_args

    specificity = tn / (tn+fp)

    fold_quantity = 10
    cross_validation_score = cross_val_score(classifier, x_test, y_test, cv=fold_quantity, scoring='accuracy'),
    cross_validation_mean = np.mean(cross_validation_score)
    cross_validation_std = np.std(cross_validation_score)

    return (
      accuracy, 
      precision, 
      recall, 
      f1, 
      specificity,
      cross_validation_score, 
      cross_validation_mean,
      cross_validation_std,
      confusion_matrix_args
    )


def print_metrics(x_tes, y_test, y_pred, classifier, metrics):
  accuracy, precision, recall, f1, specificity, cross_validation_score, cross_validation_mean, cross_validation_std, confusion_matrix_args = metrics
  
  confusion_matrix_args = metrics[-1]

  np.set_printoptions(precision=2)
  confusion_matrix_display = ConfusionMatrixDisplay.from_estimator(
      classifier,
      x_test,
      y_test,
      display_labels=LABELS,
      cmap=plt.cm.Blues,
      normalize='true' if NORMALIZE_CONFUSION_MATRIX else None,
  )
  title = "Normalized confusion matrix"

  confusion_matrix_display.ax_.set_title(title)

  if VERBOSE:
    tn, fp, fn, tp = confusion_matrix_args
    print('-------------------------------------------')
    print(f'Labels: {LABELS}')
    print(f'Positive label: {POSITIVE_LABEL}')
    print(f'Negative label: {NEGATIVE_LABEL}')
    print('----------------------------')
    print(f'accuracy: {round(accuracy, 2)}')
    print(f'precision: {round(precision, 2)}')
    print(f'recall: {round(recall, 2)}')
    print(f'f1: {round(f1, 2)}')
    print(f'specificity: {round(specificity, 2)}')
    print('----------------------------')
    print(f'cross_validation_score: {cross_validation_score}')
    print(f'cross_validation_mean: {round(cross_validation_mean,2)}')
    print(f'cross_validation_std: {round(cross_validation_std, 2)}')
    print('----------------------------')
    print(f'true negative: {tn}, false positive: {fp}, false negative: {fn}, true positive: {tp}')
    print(f'confusion_matrix_args: {confusion_matrix_args}')
    print('-------------------------------------------')


  if SHOW_MATPLOTLIB_OUTPUT:
    plt.show()

dataset = read_dataset()
x, y = preprocess_dataset(dataset)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=SHUFFLE, test_size=TEST_SIZE)

classifier = read_saved_classifier()
# se não existir um classifier previamente treinado, executa a rotina de treino
if classifier is None:
    if VERBOSE:
      print('no saved classifier found')
    classifier = run_training(x_train, y_train)
    save_classifier(classifier)
else:
  if VERBOSE:
    print(f'using exising classifier... ({CLASSIFIER_FILE})')

if VERBOSE:
  print('testing...')
y_pred = classifier.predict(x_test)
metrics = build_metrics(x_test, y_test, y_pred, classifier)
print_metrics(x_test, y_test, y_pred, classifier, metrics)
