import pickle
def save_model(model, filename="RfModel"):
    """
    Saves a trained model as a binary file using pickle.

    Parameters:
    -----------
    model : object
        A trained model object to be saved.
    filename : str, optional (default="RfModel")
        The filename to be used when saving the model.

    Returns:
    --------
    None
    """

    # Save the model object to a file using pickle
    with open("../models/" + filename, 'wb') as file:
        pickle.dump(model, file)

    # Print confirmation message
    print("Model saved as", filename)
