import time
import os
from config.load_config import load_config
from data.read_and_preprocess_data import read_and_preprocess_data
from evaluate.evaluate_model_mean import evaluate_model_mean
from crossValidation.run_cv import run_cv

if __name__ == "__main__":
    t0 = time.time()

    config = load_config("my_configuration.yaml")

    X, y, df = read_and_preprocess_data(os.path.join(config["data_directory"], config["data_name"]))

    lst_acc, lst_auc, lst_f1, lst_mcc = run_cv(X, y, config)

    evaluate_model_mean(lst_acc, lst_auc, lst_mcc, lst_f1, time.time() - t0, 0)
