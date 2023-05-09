import os
import time
import numpy as np
from config.load_config import load_config
from evaluate.evaluate_model_mean import evaluate_model_mean
from evaluate.evaluate_model import evaluate_model
from features.filters import filters
from cross_validation import cross_validation
from features.scaling_data import scaler
from data.read_and_preprocess_data import read_and_preprocess_data
from models.train_model import train_random_forest
from models.predict_model import predict_model
from features.measuring_stability_of_FS import calculate_asm


if __name__ == "__main__":
    # Wczytywanie konfiguracji
    config = load_config("my_configuration.yaml")

    # Tworzenie klasyfikatora

    # Wczytywanie danych
    X, y, df = read_and_preprocess_data(
        os.path.join(config["data_directory"], config["data_name"]),
        config["target_name"],
        config["stop_features"],
    )

    # Feature Scaling for input features.
    x_scaled = scaler(X)

    asm_features = list()

    acc_list = []
    auc_list = []
    mcc_list = []
    f1_list = []

    x_train_fold = filters(
        X, y, config["n_features"], config["filter"], df
    )

    print(x_train_fold)

    train_indices_list, test_indices_list = cross_validation(
        x_scaled, y, n_splits=3, random_state=1
    )

