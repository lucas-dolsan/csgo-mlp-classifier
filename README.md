## Problema
O problema deste trabalho trata-se da **classificação** do resultado de **uma rodada** de uma partida de **Counter Strike: Global Offensive**, sendo vitória do time "*Counter-Terrorists*" (CT) ou vitória do time "*Terrorists*" (TR).


## Dataset
O dataset escolhido para o desenvolvimento deste classificador foi retirado do [Kaggle](kaggle.com),
e é constituído em diversos _snapshots_ do estado atual de uma partida de **Counter Strike: Global Offensive.**

Este é um screenshot de uma rodada em andamento.

![Rodada de CS:GO em andamento](screenshots/round-screenshot.jpeg)

Uma partida de **CS:GO** resulta num arquivo local na máquina de cada um dos 10 jogadores.
O arquivo é um `.dem` (demo), que pode ser aberto em uma instância do jogo para assistir a partida jogada, ou exportado para ferramentas de terceiros para realizar _parsing_.

Existem diversas informações relevantes para determinar a qual time possui maior chance de vitória, aqui são algumas das colunas extraídas de uma rodada de uma partida.

![Dados extraídos do dataset](screenshots/dataset-screenshot.jpeg)

Disponível em: <https://www.kaggle.com/christianlillelund/csgo-round-winner-classification>

### Técnica
A técnica utilizada foi a de um **classificador**, implementado com uma estrutura **Multi-Layer Perceptron**, utilizando a biblioteca para aprendizado de _machine learning_ **Scikit Learn**, implementada em **Python**.

O objetivo do **classificador** é conseguir indicar qual dos dois times (CT, TR) será o vencedor de uma rodada em questão, ou seja, o **classificador** desenvolvido nesse projeto é do tipo *binário*.

Existem inúmeras vantagens de se usar o Scikit Learn, especialmente para projetos que tem como objetivo o aprendizado na área de _machine learning_, pois existem inúmeras funções pré-implementadas, para pré-processamento, treinamento e métricas de avaliação, sem falar dos datasets incluídos diretamente na biblioteca.

A estrutura da rede em questão segue o seguinte modelo:

![ANN MLP Binary Classifier](screenshots/mlp-binary-classifier.png)

A camada de entrada possui um neurônio para cada característica de um registro do dataset, ou seja, para cada coluna do dataset, portanto são 97 neurônios de entrada.

A camada oculta de um MLP pode possuir uma quantidade arbitrária de camadas, e da mesma forma, cada camada pode possuir uma quantidade qualquer de neurônios.

A camada de saída também pode possuir uma quantidade arbitrária de neurônios, porém, no caso dessa implementação, pelo fato do objetivo da rede ser um classificador binário, essa camada possuirá somente dois neurônios.

### Métricas de avaliação

### Resultados obtidos

      medium_dataset.csv
      v5.pkl
      max_iter:10925
      batch_size:auto
      hidden_layer_sizes:120
      accuracy: 0.91
      precision: 0.91
      recall: 0.9
      f1: 0.9
      specificity: 0.9
      cross_validation_score:
         (array([0.84, 0.83, 0.92, 0.83, 0.86, 0.79, 0.87, 0.81, 0.78, 0.83]),)
      cross_validation_mean: 0.84
      cross_validation_std: 0.04

### Instruções de uso do software

Instale as dependências:
  - Instale o python (versão utilizada durante o desenvolvimento: **Python 3.9.2**)
  - Instale as bibliotecas utilizadas:
      ```pip install pandas numpy matplotlib sklearn```          

Para treinar/testar um classificador, execute o comando:
```python3 classifier.py``` 
#### Importante:

Essas variáveis estão definidas no topo do arquivo **classifier.py**:

 - ```CLASSIFIER_FILE```

    define o nome do arquivo para ler/gravar o classificador, se não existir um arquivo
    com o nome especificado dentro da pasta ```trained-classifiers/```, o script irá treinar um classificador com os parâmetros definidos no **classifier.py**. Se houver um arquivo com o nome especificado, o script não irá treinar novamente, o classficador será serializado de volta em memória, e o script irá direto para a etapa de testes e construção dos resultados das métricas de avaliação.

    **Dica:** Você pode redirecionar o *stdout* da execução do **classifier.py** para um arquivo de texto, para armazenar quais parâmetros foram usados no treinamento do classificador, e quais foram os resultados das métricas.

    eg.: ```python3 classifiers.py > output.txt```


- ```DATASET_FILE```
    Define qual o arquivo será utilizado como *dataset*, atualmente existem três versões diferentes do *dataset* original baixado do **Kaggle**:
     - ```small_dataset.csv```: Uma versão curta do *dataset* original;
     - ```medium_dataset.csv```: Uma versão média do *dataset* original;
     - ```full_dataset.csv```: O Próprio *dataset* original;

### Vídeo

### Referências