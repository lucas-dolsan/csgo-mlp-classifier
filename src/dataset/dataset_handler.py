from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import config

class DatasetHandler:

  # Lê o dataset em formato csv, com delimiter=, na pasta de datasets do projeto
  def read_dataset():
      if config.VERBOSE:
        print(f'reading dataset... ({config.DATASET_FILE})')
      filepath = f'datasets/{config.DATASET_FILE}'
      return pd.read_csv(filepath, delimiter=',')

  # Pré-processa o dataset, preparando os dados para treinamento/testes
  def preprocess_dataset(dataset):
      if config.VERBOSE:
        print('preprocessing dataset...')
      # Precisamos usar o LabelEncoder para mapear a coluna 'map' para valores numéricos
      # ex.: {'de_dust2' -> 01; 'de_mirage' -> 02; etc}

      label_encoder = LabelEncoder()
      label_encoder.fit(dataset['map'])
      dataset['map'] = label_encoder.transform(dataset['map'])

      # para definir x, precisamos  ler todas as colunas, menos a última
      # pois a última coluna ('round_winner') armazena o resultado de cada rodada
      x = dataset[dataset.columns[:-1]]

      x = StandardScaler().fit_transform(x)

      # resultado de qual time (CT ou TR) venceu a rodada
      y = dataset['round_winner']

      return x, y

  def organize_dataset(x, y):
    return train_test_split(
      x,
      y,
      shuffle=config.SHUFFLE_DATASET,
      test_size=config.DATASET_TESTING_SIZE
    )
