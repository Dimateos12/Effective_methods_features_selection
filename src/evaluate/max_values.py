import csv
from config.load_config import load_config

config = load_config("my_configuration.yaml")


def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data.append(row)
    return data


def find_best_performance(data):
    best_acc = max(data, key=lambda x: x['ACC'])
    best_auc = max(data, key=lambda x: x['AUC'])
    best_mcc = max(data, key=lambda x: x['MCC'])
    best_f1 = max(data, key=lambda x: x['F1'])

    best_acc_value = best_acc['ACC']
    best_auc_value = best_auc['AUC']
    best_mcc_value = best_mcc['MCC']
    best_f1_value = best_f1['F1']

    best_acc_features = best_acc['Number of Features']
    best_auc_features = best_auc['Number of Features']
    best_mcc_features = best_mcc['Number of Features']
    best_f1_features = best_f1['Number of Features']

    return {
        'ACC': best_acc_value,
        'AUC': best_auc_value,
        'MCC': best_mcc_value,
        'F1': best_f1_value,
        'ACC Features': best_acc_features,
        'AUC Features': best_auc_features,
        'MCC Features': best_mcc_features,
        'F1 Features': best_f1_features
    }


def max_values():
    csv_file1 = config['scores_file_Mrmr']
    csv_file2 = config['scores_file_ReliefF']
    csv_file3 = config['scores_file_U-test']

    data1 = read_csv(csv_file1)
    data2 = read_csv(csv_file2)
    data3 = read_csv(csv_file3)

    best_performance1 = find_best_performance(data1)
    best_performance2 = find_best_performance(data2)
    best_performance3 = find_best_performance(data3)

    print("Najlepsze wyniki dla pliku MRMR:")
    print(best_performance1)

    print("Najlepsze wyniki dla pliku ReliefF:")
    print(best_performance2)

    print("Najlepsze wyniki dla pliku U-test:")
    print(best_performance3)

