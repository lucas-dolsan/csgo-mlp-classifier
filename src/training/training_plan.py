from training.training_parameters import parameters

# define uma fila combinações de parametros que devem ser treinadas
# todas as combinações herdam os parametros definidos em training_parameters
training_plan = [
  {
    **parameters,
    'hidden_layer_sizes': (30)
  },
  {
    **parameters,
    'hidden_layer_sizes': (180)
  },
  {
    **parameters,
    'hidden_layer_sizes': (60)
  },
  {
    **parameters,
    'hidden_layer_sizes': (120)
  },
  {
    **parameters,
    'hidden_layer_sizes': (90)
  },
  {
    **parameters,
    'hidden_layer_sizes': (100)
  },
]