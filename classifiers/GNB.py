from sklearn.naive_bayes import GaussianNB

def get_gaussian_nb_classifier(var_smoothing=1e-9):
    """
    Returns an initialized Gaussian Naive Bayes classifier instance
    with a tuned var_smoothing parameter for acoustic feature stability.
    """
    return GaussianNB(var_smoothing=var_smoothing)