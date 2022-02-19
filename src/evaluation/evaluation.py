from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neural_network import MLPClassifier
import config

def evaluate_classifier(x_test, y_test, classifier: MLPClassifier):
  if config.VERBOSE:
    print('building metrics...')

  y_pred = classifier.predict(x_test)
    
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average="weighted")
  recall = recall_score(y_test, y_pred, average='binary', pos_label=config.POSITIVE_LABEL)
  f1 = f1_score(y_test, y_pred, average='binary', pos_label=config.POSITIVE_LABEL)

  confusion_matrix_args = confusion_matrix(y_test, y_pred).ravel()

  tn, fp, fn, tp = confusion_matrix_args

  specificity = tn / (tn+fp)

  cross_validation_score = cross_val_score(
    classifier,
    x_test,
    y_test,
    cv=config.CROSS_VAL_K_FOLD,
    scoring='accuracy'
  )
  
  cross_validation_mean = np.mean(cross_validation_score)
  cross_validation_std = np.std(cross_validation_score)

  return {
    'accuracy': round(accuracy, 2),
    'precision': round(precision, 2),
    'recall': round(recall, 2),
    'f1': round(f1, 2), 
    'specificity': round(specificity, 2),
    'cross_validation_score': cross_validation_score.tolist(), 
    'cross_validation_mean': round(cross_validation_mean, 2),
    'cross_validation_std': round(cross_validation_std, 2),
    'confusion_matrix_args': confusion_matrix_args.tolist()
  }