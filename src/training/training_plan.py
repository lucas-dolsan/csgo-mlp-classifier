from training.training_parameters import parameters

# define uma fila combinações de parametros que devem ser treinadas
# todas as combinações herdam os parametros definidos em training_parameters
training_plan = [
  {
    "criterion": "gini",
    "min_samples_leaf": 5,
    "min_samples_split": 5,
  },
  {
    "criterion": "poisson",
    "min_samples_leaf": 5,
    "min_samples_split": 5,
  },
  {
    "criterion": "poisson",
    "min_samples_leaf": 10,
    "min_samples_split": 10,
  },
  {
    "criterion": "gini",
    "min_samples_leaf": 10,
    "min_samples_split": 5,
  },
  {
    "criterion": "poisson",
    "min_samples_leaf": 5,
    "min_samples_split": 10,
  }
]