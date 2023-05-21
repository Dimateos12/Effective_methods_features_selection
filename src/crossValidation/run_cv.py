from crossValidation.cross_validation import cross_validation
from crossValidation.get_score import get_scores


def run_cv(X, y, config, ):
    """
    Runs cross-validation for the given dataset and configuration.
    Args:
    X (numpy.ndarray): The feature matrix of the dataset.
    y (numpy.ndarray): The target array of the dataset.
    config (dict): The configuration dictionary containing the parameters
        for the cross-validation.

    Returns:
        Four lists containing the accuracy, AUC, F1, and MCC scores for each fold
        and repeat of the cross-validation.

    """

    lst_acc = []
    lst_auc = []
    lst_f1 = []
    lst_mcc = []
    lst_amc = []

    def run_fold(X_train, X_test, y_train, y_test):
        """
            Runs the training and testing for a single fold of the cross-validation.

            Args:
                X_train (numpy.ndarray): The feature matrix for the training set.
                X_test (numpy.ndarray): The feature matrix for the testing set.
                y_train (numpy.ndarray): The target array for the training set.
                y_test (numpy.ndarray): The target array for the testing set.
        """

        scores = get_scores(X_train, X_test, y_train, y_test)
        lst_acc.append(scores[0])
        lst_auc.append(scores[1])
        lst_f1.append(scores[2])
        lst_mcc.append(scores[3])
        lst_amc.append(scores[4])

    def run_repeat():
        """
            Runs the cross-validation for the specified number of repeats and folds.
        """
        folds = cross_validation()
        for train_index, test_index in folds.split(X, y):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], \
                y[train_index], y[test_index]
            run_fold(X_train, X_test, y_train, y_test)

    for i in range(config['repeat']):
        print(f"Repeat: {i + 1}")
        run_repeat()

    return lst_acc, lst_auc, lst_f1, lst_mcc, lst_amc
