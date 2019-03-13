import build_dataset
import numpy as np
import pandas as pd
from classifiers import logistic_model, knn_model, mlp_model, svm_model
from __init__ import *
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')


def visualization(name, classifier_result_name, classifier_result_acc):
    plt.cla()
    plt.clf()
    acc = pd.Series.from_array(classifier_result_acc)
    plt.figure()
    max_acc = max(classifier_result_acc)
    ax = acc.plot(kind='bar', edgecolor='black', width=1.0, color=['red' if row == max_acc else "orange" for row in classifier_result_acc])
    ax.set_title(name)
    ax.set_ylabel('Accuracy')
    ax.set_yticks(np.arange(0.0, 110.0, 10.0))
    ax.set_xticklabels(classifier_result_name, rotation=360)
    labels = ['%.2f' % elem for elem in classifier_result_acc]
    boxes = ax.patches
    for box, label in zip(boxes, labels):
        height = box.get_height()
        ax.text(box.get_x() + box.get_width() / 2, height + 0.1, label, ha='center', va='bottom')
    plt.axis('tight')
    figure_path = os.path.join(result_location, name+".png")
    plt.savefig(figure_path, format="png")

def main():
    """
    Execute generic classification methods on DNA methylation data
    """
    logger.info("We work on "+attribute+" classification")
    train, test = build_dataset.build()
    classifier_result_name = []
    classifier_result_acc = []

    logger.info('Applying Logistic Regression')
    name, acc = logistic_model(train, test, "l1")
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = logistic_model(train, test, "l2")
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    visualization("LR", classifier_result_name, classifier_result_acc)

    classifier_result_name.clear()
    classifier_result_acc.clear()
    logger.info('Applying KNN')
    name, acc = knn_model(train, test, 2)
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = knn_model(train, test, 3)
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = knn_model(train, test, 4)
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = knn_model(train, test, 5)
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    visualization("KNN", classifier_result_name, classifier_result_acc)

    classifier_result_name.clear()
    classifier_result_acc.clear()
    logger.info('Applying SVM')
    name, acc = svm_model(train, test, "linear")
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = svm_model(train, test, "rbf")
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = svm_model(train, test, "poly", 2)
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = svm_model(train, test, "poly", 3)
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = svm_model(train, test, "poly", 4)
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    visualization("SVM", classifier_result_name, classifier_result_acc)

    classifier_result_name.clear()
    classifier_result_acc.clear()
    logger.info('Applying MLP')
    name, acc = mlp_model(train, test, "logistic", (100,))
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = mlp_model(train, test, "logistic", (100, 100))
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = mlp_model(train, test, "tanh", (100,))
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = mlp_model(train, test, "tanh", (100, 100))
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = mlp_model(train, test, "relu", (100,))
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    name, acc = mlp_model(train, test, "relu", (100, 100))
    classifier_result_name.append(name)
    classifier_result_acc.append(acc)
    visualization("MLP", classifier_result_name, classifier_result_acc)


if __name__ == '__main__':
    main()
