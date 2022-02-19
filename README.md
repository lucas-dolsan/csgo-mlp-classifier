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

A camada de entrada possui um neurônio para cada característica de um registro do dataset, ou seja, para cada coluna do dataset, portanto são 96 neurônios de entrada.

### Métricas de avaliação
        accuracy: Acurácia
        precision: Precisão
        recall: Recall
        f1: F1 Score
        specificity: Especificidade
        cross_validation_score: Validação cruzada
        cross_validation_mean: Média da validação cruzada
        cross_validation_std: Desvio padrão da validação cruzada
        confusion_matrix_args: Matriz de confusão

### Resultados obtidos
Foram realizados diversos experimentos, modificando uma série de parâmetros diferentes.
O treinamento foi baseado em um conjunto de parâmetros, a maior parte deles são os valores padrões do **MLPClassifier** do **SKLearn**.

Os experimentos realizados foram feitos em grupos, nessa sequência:

- **Camadas ocultas**
- **Funções de ativação**
- **Otimizações**
- **Tamanhos de lote**

Ao finalizar um grupo de testes, a configuração combinação de parâmetros que resultou
nas melhores métricas, será preservada ao avançar para o próximo grupo de testes.

#### Melhor resultado (v14-full_dataset)

A melhor versão encontrada, entre todos os grupos de teste, foi a **v14-full_dataset**.

Essa configuração está no primeiro grupo de testes, todos os testes realizados posteriormente,

**não resultaram em nenhuma melhoria significativa**.

#### Camadas ocultas:
Configurações utilizadas:
 - Uma camada oculta, com 30 neurônios; **(v0-full_dataset)**;
```
"accuracy": 0.77,
"precision": 0.77,
"recall": 0.79,
"f1": 0.77,
"specificity": 0.79,
"cross_validation_score": [
    0.75, 0.76, 0.75,
    0.75, 0.75, 0.76,
    0.75, 0.75, 0.76,
    0.75
],
"cross_validation_mean": 0.76,
"cross_validation_std": 0.0,
"confusion_matrix_args": [
    [15657, 4115],
    [5068, 15556]
]
```
 - Uma camada oculta, com 180 neurônios; **(v1-full_dataset)**;
```
"accuracy": 0.81,
"precision": 0.81,
"recall": 0.81,
"f1": 0.81,
"specificity": 0.81,
"cross_validation_score": [
    0.77, 0.78, 0.76,
    0.77, 0.78, 0.76,
    0.76, 0.78, 0.76,
    0.77
],
"cross_validation_mean": 0.77,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [15962, 3818],
    [3735, 16881]
]
```
 - Uma camada oculta, com 60 neurônios; **(v2-full_dataset)**;
```
"accuracy": 0.79,
"precision": 0.79,
"recall": 0.8,
"f1": 0.79,
"specificity": 0.8,
"cross_validation_score": [
    0.76, 0.76, 0.75,
    0.75, 0.76, 0.76,
    0.75, 0.77, 0.75,
    0.76
],
"cross_validation_mean": 0.76,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [15891, 3864],
    [4623, 16018]
]
```
 - Uma camada oculta, com 120 neurônios; **(v3-full_dataset)**;
```
"accuracy": 0.8,
"precision": 0.8,
"recall": 0.78,
"f1": 0.79,
"specificity": 0.78,
"cross_validation_score": [
    0.77, 0.77, 0.78,
    0.77, 0.77, 0.76,
    0.77, 0.76, 0.76,
    0.77
],
"cross_validation_mean": 0.77,
"cross_validation_std": 0.0,
"confusion_matrix_args": [
    [15379, 4434],
    [3679, 16904]
]
```
 - Uma camada oculta, com 90 neurônios; **(v4-full_dataset)**;
```
"accuracy": 0.79,
"precision": 0.79,
"recall": 0.8,
"f1": 0.79,
"specificity": 0.8,
"cross_validation_score": [
    0.76, 0.77, 0.76,
    0.77, 0.76, 0.76,
    0.76, 0.77, 0.77,
    0.75
],
"cross_validation_mean": 0.77,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [15931, 3939],
    [4353, 16173]
]
```
 - Uma camada oculta, com 180 neurônios; **(v5-full_dataset)**;
```
"accuracy": 0.79,
"precision": 0.8,
"recall": 0.83,
"f1": 0.8,
"specificity": 0.83,
"cross_validation_score": [
    0.76, 0.76, 0.77,
    0.76, 0.76, 0.76,
    0.77, 0.76, 0.78,
    0.76
],
"cross_validation_mean": 0.77,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16416, 3333],
    [5027, 15620]
]
```
 - Uma camada oculta, com 100 neurônios; **(v6-full_dataset)**;
```
"accuracy": 0.78,
"precision": 0.78,
"recall": 0.79,
"f1": 0.78,
"specificity": 0.79,
"cross_validation_score": [
    0.74, 0.75, 0.76,
    0.74, 0.76, 0.75,
    0.75, 0.74, 0.76,
    0.75
],
"cross_validation_mean": 0.76,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [15589, 4194],
    [4722, 15891]
]
```
 - Duas camadas ocultas, com 30 e 30 neurônios respectivamente; **(v7-full_dataset)**;
```
"accuracy": 0.83,
"precision": 0.83,
"recall": 0.82,
"f1": 0.83,
"specificity": 0.82,
"cross_validation_score": [
    0.77, 0.78, 0.78,
    0.77, 0.78, 0.77, 0.78,
    0.77, 0.79, 0.79
],
"cross_validation_mean": 0.78,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16190, 3604],
    [3229, 17373]
]
```
 - Duas camadas ocultas, com 180 e 30 neurônios respectivamente; **(v8-full_dataset)**;
```
"accuracy": 0.8,
"precision": 0.8,
"recall": 0.8,
"f1": 0.8,
"specificity": 0.8,
"cross_validation_score": [
    0.76, 0.76, 0.76,
    0.76, 0.76, 0.75,
    0.76, 0.75, 0.76,
    0.76
],
"cross_validation_mean": 0.76,
"cross_validation_std": 0.0,
"confusion_matrix_args": [
    [15800, 3933],
    [4008, 16655]
]
```
 - Duas camadas ocultas, com 120 e 30 neurônios respectivamente; **(v9-full_dataset)**;
```
"accuracy": 0.81,
"precision": 0.82,
"recall": 0.85,
"f1": 0.82,
"specificity": 0.85,
"cross_validation_score": [
    0.76, 0.78, 0.77, 0.775,
    0.77, 0.77, 0.77,
    0.77, 0.76, 0.76
],
"cross_validation_mean": 0.77,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16928, 2980],
    [4543, 15945]
]
```
 - Duas camadas ocultas, com 90 e 30 neurônios respectivamente; **(v10-full_dataset)**;
```
"accuracy": 0.81,
"precision": 0.81,
"recall": 0.79,
"f1": 0.8,
"specificity": 0.79,
"cross_validation_score": [
    0.76, 0.76, 0.77,
    0.77, 0.76, 0.77,
    0.77, 0.76, 0.76,
    0.77
],
"cross_validation_mean": 0.77,
"cross_validation_std": 0.0,
"confusion_matrix_args": [
    [15533, 4138],
    [3689, 17036]
]
```
 - Duas camadas ocultas, com 100 e 30 neurônios respectivamente; **(v11-full_dataset)**;
```
"accuracy": 0.81,
"precision": 0.81,
"recall": 0.81,
"f1": 0.81,
"specificity": 0.81,
"cross_validation_score": [
    0.77, 0.76, 0.75,
    0.76, 0.77, 0.76,
    0.78, 0.77, 0.78,
    0.76
],
"cross_validation_mean": 0.77,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16082, 3869],
    [3786, 16659]
]
```
 - Duas camadas ocultas, com 90 e 30 neurônios respectivamente; **(v12-full_dataset)**;
```
"accuracy": 0.81,
"precision": 0.82,
"recall": 0.76,
"f1": 0.8,
"specificity": 0.76,
"cross_validation_score": [
    0.77, 0.77, 0.78,
    0.78, 0.77, 0.78,
    0.77, 0.78, 0.79,
    0.78
],
"cross_validation_mean": 0.78,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [15059, 4857],
    [2692, 17788]
]
```
 - Duas camadas ocultas, com 150 e 50 neurônios respectivamente; **(v13-full_dataset)**;
```
"accuracy": 0.83,
"precision": 0.83,
"recall": 0.84,
"f1": 0.83,
"specificity": 0.84,
"cross_validation_score": [
    0.78, 0.80, 0.77,
    0.78, 0.79, 0.77,
    0.79, 0.77, 0.78,
    0.78
],
"cross_validation_mean": 0.79,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16710, 3134],
    [3605, 16947]
]
```
 - Duas camadas ocultas, com 150 e 80 neurônios respectivamente; **(v14-full_dataset)**;

Esta foi a melhor configuração encontrada.
```
"accuracy": 0.84,
"precision": 0.84,
"recall": 0.84,
"f1": 0.84,
"specificity": 0.84,
"cross_validation_score": [
    0.79, 0.78, 0.78,
    0.78, 0.79, 0.79,
    0.79, 0.77, 0.77,
    0.79
],
"cross_validation_mean": 0.79,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16654, 3077],
    [3422, 17243]
]
```
 - Duas camadas ocultas, com 120 e 30 neurônios respectivamente; **(v15-full_dataset)**;
```
"accuracy": 0.82,
"precision": 0.82,
"recall": 0.82,
"f1": 0.81,
"specificity": 0.82,
"cross_validation_score": [
    0.78, 0.77, 0.78,
    0.77, 0.77, 0.76,
    0.78, 0.78, 0.75,
    0.77
],
"cross_validation_mean": 0.78,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16221, 3463],
    [3945, 16767]
]
```
 - Duas camadas ocultas, com 120 e 50 neurônios respectivamente; **(v16-full_dataset)**;
```
"accuracy": 0.82,
"precision": 0.82,
"recall": 0.82,
"f1": 0.82,
"specificity": 0.82,
"cross_validation_score": [
    0.77, 0.78, 0.77,
    0.76, 0.77, 0.79,
    0.79, 0.77, 0.77,
    0.77
],
"cross_validation_mean": 0.78,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16419, 3488],
    [3652, 16837]
]
```
 - Duas camadas ocultas, com 165 e 45 neurônios respectivamente; **(v17-full_dataset)**;
```
"accuracy": 0.84,
"precision": 0.84,
"recall": 0.84,
"f1": 0.83,
"specificity": 0.84,
"cross_validation_score": [
    0.78, 0.78, 0.79,
    0.78, 0.78, 0.77,
    0.78, 0.78, 0.77,
    0.78
],
"cross_validation_mean": 0.78,
"cross_validation_std": 0.01,
"confusion_matrix_args": [
    [16483, 3229],
    [3288, 17396]
]
```

**A melhor configuração encontrada no grupo de testes anterior, será definida como base para o próximo grupo de testes**

#### Funções de ativação
  - Tangente hiperbólica **tanh**
  
    ```f(x) = tanh(x)```
    ```
    "accuracy": 0.83,
    "precision": 0.83,
    "recall": 0.82,
    "f1": 0.83,
    "specificity": 0.82,
    "cross_validation_score": [
        0.78, 0.79, 0.79,
        0.79, 0.78, 0.79,
        0.79, 0.78, 0.78,
        0.79
    ],
    "cross_validation_mean": 0.79,
    "cross_validation_std": 0.0,
    "confusion_matrix_args": [
        [16226, 3541],
        [3330, 17299]
    ]
    ```

  - Função identidade (sem ativação) **identity**

    ```f(x) = x```
    ```
    "accuracy": 0.75,
    "precision": 0.75,
    "recall": 0.75,
    "f1": 0.75,
    "specificity": 0.75,
    "cross_validation_score": [
        0.74, 0.74, 0.74,
        0.74, 0.74, 0.75,
        0.75, 0.74, 0.74,
        0.76
    ],
    "cross_validation_mean": 0.75,
    "cross_validation_std": 0.01,
    "confusion_matrix_args": [
        [14903, 4859],
        [5253, 15381]
    ]
    ```
  - Função sigmoide **logistic**
  
    ```f(x) = 1 / (1 + exp(-x))```
    ```
    "accuracy": 0.83,
    "precision": 0.83,
    "recall": 0.83,
    "f1": 0.82,
    "specificity": 0.83,
    "cross_validation_score": [
        0.77, 0.77, 0.76,
        0.77, 0.76, 0.76,
        0.75, 0.76, 0.76,
        0.76
    ],
    "cross_validation_mean": 0.77,
    "cross_validation_std": 0.01,
    "confusion_matrix_args": [
        [16453, 3383],
        [3602, 16958]
    ]
    ```
  - ReLU **relu**

    ```f(x) = max(0, x)```
    ```
    "accuracy": 0.84,
    "precision": 0.84,
    "recall": 0.84,
    "f1": 0.84,
    "specificity": 0.84,
    "cross_validation_score": [
        0.79, 0.78, 0.78,
        0.78, 0.79, 0.79,
        0.79, 0.77, 0.77,
        0.79
    ],
    "cross_validation_mean": 0.79,
    "cross_validation_std": 0.01,
    "confusion_matrix_args": [
        [16654, 3077],
        [3422, 17243]
    ]
    ```
    Em todos os testes realizados, marginalmente, o melhor resultado foi com a função reLU,
    que é a função de ativação padrão, utilizada anteriormente.

#### Otimizações
#### Tamanhos de lote

#### Experimentos

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

    eg.: ```python3 ./src/classifiers.py > output.txt```


- ```DATASET_FILE```
    Define qual o arquivo será utilizado como *dataset*, atualmente existem três versões diferentes do *dataset* original baixado do **Kaggle**:
     - ```small_dataset.csv```: Uma versão curta do *dataset* original;
     - ```medium_dataset.csv```: Uma versão média do *dataset* original;
     - ```full_dataset.csv```: O Próprio *dataset* original;

### Vídeo

### Referências