from sklearn.metrics import confusion_matrix
import numpy as np

def metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tn + tp) / np.sum((tn, fp, fn, tp))
    return accuracy
