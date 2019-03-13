import pandas as pd
from __init__ import *
from sklearn import metrics, linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def calculate_metrics(true_values, predicted_values):
    """
    Calculate precision, recall, f1-score and support based on classifier output
    """
    acc = metrics.accuracy_score(y_true=true_values, y_pred=predicted_values) * 100
    report = metrics.classification_report(y_true=true_values, y_pred=predicted_values)
    result = "Accuracy = " + str(acc) + "\n" + str(report)
    logger.info(result)
    return acc


def svm_model(train, test, kernel, degree=None):
    if kernel == "poly":
        clf = svm.SVC(kernel=kernel, degree=degree)
    else:
        clf = svm.SVC(kernel=kernel)
    clf.fit(X=train[train.columns[:-1]], y=train[attribute])
    predicted = clf.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)
    if kernel == "poly":
        name = "{0} {1}".format(kernel, degree)
    else:
        name = "{0}".format(kernel)
    logger.info(name)
    return name, calculate_metrics(test[attribute], predicted)


def knn_model(train, test, neighbors):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X=train[train.columns[:-1]], y=train[attribute])
    predicted = knn.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)
    name = "n={0}".format(neighbors)
    logger.info(name)
    return name, calculate_metrics(test[attribute], predicted)


def mlp_model(train, test, activation, hidden_layers):
    mlp = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layers)
    mlp.fit(X=train[train.columns[:-1]], y=train[attribute])
    predicted = mlp.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)
    if activation == "logistic":
        activation = "sigmoid"
    name = "{0}\n{1} layer".format(activation, len(hidden_layers))
    logger.info(name)
    return name, calculate_metrics(test[attribute], predicted)


def logistic_model(train, test, penalty):
    log_model = linear_model.LogisticRegression(penalty=penalty)
    log_model.fit(X=train[train.columns[:-1]], y=train[attribute])
    predicted = log_model.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)
    name = "{0} penalty".format(penalty)
    logger.info(name)
    return name, calculate_metrics(test[attribute], predicted)

