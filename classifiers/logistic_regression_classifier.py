from sklearn.linear_model import LogisticRegression

def get_logistic_regression_classifier(C=1.0, max_iter=1000, random_state=42):
    """
    Returns an initialized Logistic Regression classifier instance.
    The 'multi_class' parameter is omitted as 'multinomial' is now the standard default behavior.
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver='lbfgs',
        random_state=random_state
    )