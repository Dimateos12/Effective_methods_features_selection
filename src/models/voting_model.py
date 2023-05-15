from sklearn.ensemble import VotingClassifier

def create_voting_model(models):
    """
    Creates a voting model from a list of machine learning models.

    Args:
        models (list): A list of machine learning models.

    Returns:
        VotingClassifier: A voting model that combines the predictions of the input models.
    """
    # Creating a list of tuples with model names and the models themselves
    model_tuples = [(model.__class__.__name__, model) for model in models]

    # Creating the voting model
    voting_model = VotingClassifier(estimators=model_tuples, voting='hard')

    return voting_model