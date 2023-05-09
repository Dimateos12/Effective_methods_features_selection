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

    # K krotna walidacja
    train_indices_list, test_indices_list = cross_validation(
        x_scaled, y, n_splits=config["k_fold"], random_state=1
    )

    for i in range(config["repeat"]):
        lst_acc = []
        lst_auc = []
        lst_mcc = []
        lst_f1 = []

        for train_index, test_index in zip(train_indices_list, test_indices_list):
            t0 = time.time()
            x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            x_train_fold = filters(
                x_train_fold, y_train_fold, config["n_features"], config["filter"]
            )

            asm_features.append(set(x_train_fold))
            model = train_random_forest()
            y_pred = predict_model(model, x_test_fold)

            acc, auc, mcc, f1 = evaluate_model(
                config["model_name"],
                config["filter"],
                y_test_fold,
                y_pred,
                config["name_each_result"],
            )
            lst_acc.append(acc)
            lst_auc.append(auc)
            lst_mcc.append(mcc)
            lst_f1.append(f1)

        acc_list.append(np.mean(lst_acc))
        auc_list.append(np.mean(lst_auc))
        mcc_list.append(np.mean(lst_mcc))
        f1_list.append(np.mean(lst_f1))
        calculate_asm(asm_features)
        evaluate_model_mean(
            config["model_name"],
            config["name_file_k_fold"],
            config["filter"],
            lst_acc,
            lst_auc,
            lst_mcc,
            lst_f1,
            config["n_features"],
            time.time() - t0,
        )
    calculate_asm(asm_features)
    evaluate_model_mean(
        config["model_name"],
        config["name_file_30_repeat"],
        config["filter"],
        lst_acc,
        lst_auc,
        lst_mcc,
        lst_f1,
        config["n_features"],
        time.time() - t0,
        asm_features
    )

