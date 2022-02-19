## Hidden Layer tests 

Esse diretório contém os experimentos realizados com diferentes configurações
de camadas ocultas.

## Melhor versão:
  ### v15-full-dataset

  #### Parâmetros
  Abaixo estão os parâmetros utilizados no treinamento do classificador.
  Apenas os parâmetros que estão **(MODIFICADO)**, foram alterados no teste,
  o restante são apenas os parâmetros **padrão** do **MLPClassifier** do **SKLearn** 

    "hidden_layer_sizes": [150, 80], (MODIFICADO)
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": "auto",
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "power_t": 0.5,
    "max_iter": 100000,
    "shuffle": true,
    "random_state": null,
    "tol": 0.0001,
    "verbose": false,
    "warm_start": false,
    "momentum": 0.9,
    "nesterovs_momentum": true,
    "early_stopping": false,
    "validation_fraction": 0.1,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-8,
    "n_iter_no_change": 10,
    "max_fun": 15000


  #### Métricas
    "accuracy": 0.84,
    "precision": 0.84,
    "recall": 0.84,
    "f1": 0.84,
    "specificity": 0.84,
    "cross_validation_score": [
      0.7952970297029703, 0.7844059405940594, 0.7814356435643565,
      0.7893564356435644, 0.7928217821782179, 0.7987623762376238,
      0.7967318643228521, 0.7717256746719485, 0.779896013864818,
      0.7967318643228521
    ],
    "cross_validation_mean": 0.79,
    "cross_validation_std": 0.01,
    "confusion_matrix_args": [16654, 3077, 3422, 17243]


### Observação
  Todos os testes subsequentes terão os seus parâmetros baseados na versão acima.