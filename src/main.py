import time
import os
from config.load_config import load_config
from data.read_and_preprocess_data import read_and_preprocess_data
from evaluate.evaluate_model_mean import evaluate_model_mean
from crossValidation.run_cv import run_cv
from visualization.visualize import compare_scores
from evaluate.max_values import max_values
from visualization.visualize import time_compare
#from visualization.visualize import venn_diagram
from features.filters import filters
from visualization.visualize import k_fold_plot
import numpy as np
from sklearn.datasets import fetch_openml

if __name__ == "__main__":

    t0 = time.time()
    config = load_config("my_configuration.yaml")

    if config['mode'] == 'score':
        X, y, df = read_and_preprocess_data(os.path.join(config["data_directory"], config["data_name"]))
        lst_acc, lst_auc, lst_f1, lst_mcc, lst_amc = run_cv(X, y, config)
        evaluate_model_mean(lst_acc, lst_auc, lst_mcc, lst_f1, time.time() - t0, lst_amc)

    if config['mode'] == 'test':
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist["data"], mnist["target"]
        y = y.astype(np.uint8)
        y_train = (y == 5)
        y_train = y_train.astype(int)
        lst_acc, lst_auc, lst_f1, lst_mcc, lst_amc = run_cv(X, y_train, config)
        evaluate_model_mean(lst_acc, lst_auc, lst_mcc, lst_f1, time.time() - t0, lst_amc)

    else:
        X, y, df = read_and_preprocess_data(os.path.join(config["data_directory"], config["data_name"]))
        #compare_scores()
        #time_compare(X, y)
        #time_compare()
        #max_values()
        #venn_diagram(X, y)
        #k_fold_plot()

