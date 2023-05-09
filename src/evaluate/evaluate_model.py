import csv

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


def evaluate_model(model_name, feature_selection_method, y_true, y_pred, file_name):
    """
    Calculates accuracy, AUC, MCC and F1 scores based on true and predicted values,
    and saves the results to a CSV file.

    Parameters:
        model_name (str): Name of the model.
        feature_selection_method (str): Feature selection method used.
        y_true (array-like of shape (n_samples,)): True labels.
        y_pred (array-like of shape (n_samples,)): Predicted labels.
        file_name (str): Name of the CSV file to save the results.

    Returns:
        tuple: A tuple of four floats representing accuracy, AUC, MCC, and F1 scores.
    """


    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Zapisanie wyników do pliku CSV
    with open("../reports/csv/" + file_name, mode="a", newline="") as csv_file:
        fieldnames = ["Model", "Feature Selection Method", "ACC", "AUC", "MCC", "F1"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Zapisanie nazw kolumn, jeśli plik jest pusty
        if csv_file.tell() == 0:
            writer.writeheader()

        # Zapisanie wyników
        writer.writerow(
            {
                "Model": model_name,
                "Feature Selection Method": feature_selection_method,
                "ACC": acc,
                "AUC": auc,
                "MCC": mcc,
                "F1": f1,
            }
        )

    return acc, auc, mcc, f1
