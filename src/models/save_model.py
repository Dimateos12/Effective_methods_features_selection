import pickle
def save_model(model, filename):
    # zapisanie modelu do pliku
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved as", filename)