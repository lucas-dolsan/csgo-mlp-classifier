from training.training_parameters import parameters

# define uma fila combinações de parametros que devem ser treinadas
# todas as combinações herdam os parametros definidos em training_parameters
training_plan = [
  {
    **parameters,
    "max_iter": 150000,
  }
]