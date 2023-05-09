import csv

import numpy as np


def evaluate_model_mean(
    model_name,
    file_name,
    feature_selection_method,
    lst_acc,
    lst_auc,
    lst_mcc,
    lst_f1,
    num_of_features,
    time,
    stability,
):
    """
    Calculates mean values of evaluation metrics and saves the results to a CSV file.

    Parameters:
        model_name (str): name of the machine learning model.
        file_name (str): name of the CSV file to which the results will be saved.
        feature_selection_method (str): name of the feature selection method
            used in the model.
        lst_acc (list of floats): list of accuracy scores obtained during
            cross-validation.
        lst_auc (list of floats): list of area under ROC curve scores obtained
            during cross-validation.
        lst_mcc (list of floats): list of Matthews correlation coefficient
            scores obtained during cross-validation.
        lst_f1 (list of floats): list of F1 scores
            obtained during cross-validation.
        num_of_features (int): number of features
            selected in feature selection.
        time (float): time taken to run
            cross-validation.
        stability (list of sets): measure stability
            of feature selection.

    Returns:
        None
    """
    acc_mean = np.mean(lst_acc)
    auc_mean = np.mean(lst_auc)
    mcc_mean = np.mean(lst_mcc)
    f1_mean = np.mean(lst_f1)

    # Zapisanie wyników do pliku CSV
    with open("../reports/csv/" + file_name, mode="a", newline="") as csv_file:
        fieldnames = [
            "Model",
            "Feature Selection Method",
            "ACC",
            "AUC",
            "MCC",
            "F1",
            "Number of Features",
            "Time",
            "MeasuringFS",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Zapisanie nazw kolumn, jeśli plik jest pusty
        if csv_file.tell() == 0:
            writer.writeheader()

        # Zapisanie wyników
        writer.writerow(
            {
                "Model": model_name,
                "Feature Selection Method": feature_selection_method,
                "ACC": acc_mean,
                "AUC": auc_mean,
                "MCC": mcc_mean,
                "F1": f1_mean,
                "Number of Features": num_of_features,
                "Time": time,
                "MeasuringFS": stability,
            }
        )
