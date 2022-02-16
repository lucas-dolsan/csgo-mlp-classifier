## Problema
O problema deste trabalho trata-se da classificação do resultado de uma rodada de uma partida de Counter Strike: Global Offensive, sendo vitória do time "Counter-Terrorist" ou vitória do time "Terrorist".


## Dataset
O dataset escolhido para o desenvolvimento deste classificador foi retirado do [Kaggle](kaggle.com),
e é constituído em diversos _snapshots_ do estado atual de uma partida de Counter Strike: Global Offensive.

Este é um screenshot de uma rodada em andamento.

![Rodada de CS:GO em andamento](screenshots/round-screenshot.jpeg)

Uma partida de CS:GO resulta num arquivo local na máquina de cada um dos 10 participantes.
O arquivo é um `.dem` (demo), que pode ser aberto em uma instância do jogo para assistir a partida jogada, ou exportado para ferramentas de terceiros para realizar _parsing_.

Existem diversas informações relevantes para determinar a qual time possui maior chance de vitória, aqui são algumas das colunas extraídas de uma rodada de uma partida.

![Dados extraídos do dataset](screenshots/dataset-screenshot.jpeg)

Disponível em: <https://www.kaggle.com/christianlillelund/csgo-round-winner-classification>

### Técnica
A técnica utilizada foi a de um **classificador**, implementado com uma estrutura **Multi-Layer Perceptron**, utilizando a biblioteca para aprendizado de _machine learning_ **Scikit Learn**, implementada em **Python**.

O objetivo do **classificador** é conseguir indicar qual dos dois times (CT, TR) será o vencedor de uma rodada em questão, ou seja, o **classificador** desenvolvido nesse projeto é do tipo *binário*.

Existem inúmeras vantagens de se usar o Scikit Learn, especialmente para projetos que tem como objetivo o aprendizado na área de _machine learning_, pois existem inúmeras funções pré-implementadas, para pré-processamento, treinamento e métricas de avaliação, sem falar dos datasets incluídos diretamente na biblioteca.
