from features.measuring_stability_of_FS import calculate_asm
from models.create_model import create_model
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from features.filters import filters
from config.load_config import load_config
from models.train_model import train_random_forest
import numpy as np

config = load_config("my_configuration.yaml")


def get_scores(X_train, X_test, y_train, y_test):
    model = create_model()
    X_train_before = X_train
    features_to_keep = filters(X_train, y_train, config['filter'])
    X_train = X_train[:, features_to_keep]
    X_test = X_test[:, features_to_keep]
    y_pred = train_random_forest(X_train, y_train, X_test,  model)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    asm = calculate_asm([X_train_before, X_train])
    return [acc, auc, f1, mcc, asm]
