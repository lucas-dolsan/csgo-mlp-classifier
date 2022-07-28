# arg: test_size: define a porcentagem dos dados totais do dataset que serão usados para teste
# ex.: com test_size: 0.2: 80% do dataset será usado para treinamento (x_train), e o restante (20%)
# será reservado para ser usado no teste (x_test)
#
# x_train (dados para treino)
# y_train (objetivos para treino)
# x_test (dados para teste)
# y_test (objetivos para teste)
DATASET_TESTING_SIZE=0.33
# habilita logs do progresso do script
VERBOSE=True
#se deve ou não printar os parametros usados no treinamento
SHOW_TRAINING_PARAMETERS=True

#nome do dataset a ser utilizado, contido dentro da pasta de datasets
DATASET_FILE = 'full_dataset.csv'
# Se vai embaralhar ou não os dados lidos do CSV
SHUFFLE_DATASET=True
# Se deve ou não mostrar a matriz de confusão
# numa janela do matplotlib (independente do caso o resultado sairá , stdout do script)
SHOW_MATPLOTLIB_OUTPUT=False
# Se deve ou não normalizar os dados gerados da matriz de confusão
NORMALIZE_CONFUSION_MATRIX=False

CROSS_VAL_K_FOLD=10

# definido de forma arbitrária, para fins do classificador binário
POSITIVE_LABEL='CT'
NEGATIVE_LABEL='TR'

# são as duas classes classificadas (os dois times) 
LABELS=(POSITIVE_LABEL, NEGATIVE_LABEL)