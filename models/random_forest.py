import torch

from sklearn.ensemble import RandomForestClassifier


def two_forests(M, X, y, samples_prop, max_depth, binary, output_type):
    """
    Create a collection of two random forests classifiers.

    M (int): number of tree per forest classifier.
    r (float): proportion of examples for training the first forest VS the other one.
    X, y: features and labels to learn on.
    max_samples, max_depth (ints): sklearn.ensemble.RandomForestClassifier argument.
    binary (bool): whether of not the dataset is binary.
    output_type (str): whether the trees must predict a class or a class probability for every class.

    Returns both the forests in a tuple, and the total number of base classifiers (trees, 2 * M).
    """
    assert type(M) == int, f"M must be an integer, got {M}."
    assert 0 <= samples_prop <= 1, f"samples_prop (cfg.model.samples_prop) must be in [0, 1], got {samples_prop}."
    assert max_depth == "None" or type(max_depth) == int, f"cfg.model.max_depth must be int/None, got {max_depth}."
    assert output_type in ['proba', 'class'], f"output_type must be in ['proba', 'class'], got {output_type}."

    if max_depth == "None":
        max_depth = None

    # Number of example to train the first forest on (default: have both forests to train on the same amount of data).
    m = int(len(X) * 0.5)

    # Learn one prior.
    trees1 = trained_random_forest(M, (X[:m], y[:m]), samples_prop=samples_prop, max_depth=max_depth)

    # Learn the other prior.
    trees2 = trained_random_forest(M, (X[m:], y[m:]), samples_prop=samples_prop, max_depth=max_depth)

    predictors1 = lambda x: trees_predict(x, trees1, binary_dataset=binary, output_type=output_type)
    predictors2 = lambda x: trees_predict(x, trees2, binary_dataset=binary, output_type=output_type)

    return (predictors1, predictors2), 2 * M

def trained_random_forest(M, data, samples_prop=1., max_depth=None):
    """
    Create a trained sklearn.ensemble.RandomForestClassifier instance.
    max_samples (float): proportion of data used to train each tree.
    """
    bootstrap = True
    if samples_prop == 1.:
        bootstrap = False
        samples_prop = None

    forest = RandomForestClassifier(n_estimators=M, criterion="gini", max_depth=8, bootstrap=bootstrap,
                                    max_samples=samples_prop, n_jobs=-1, class_weight='balanced_subsample')
    forest.fit(*data)

    return forest.estimators_


def trees_predict(x, trees, binary_dataset=True, output_type="class"):
    """
    Depending on the output_type, returns the prediction of each tree in the forest.
    """
    if output_type == "proba":
        pred = torch.stack([torch.from_numpy(t.predict_proba(x)).float() for t in trees], 1)
    else:
        pred = torch.stack([torch.from_numpy(t.predict(x)).float() for t in trees], 1)

    if binary_dataset:
        # If binary task, we want the predictions to be in [-1, 1].
        return 2 * pred - 1

    return pred