import build_dataset
import numpy as np
import pandas as pd
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
    # ax.set_title(name)
    ax.set_ylabel('Accuracy')
    ax.set_yticks(np.arange(0.0, 110.0, 10.0))
    ax.set_xticklabels(classifier_result_name, rotation=360)
    labels = ['%.2f' % elem for elem in classifier_result_acc]
    boxes = ax.patches
    for box, label in zip(boxes, labels):
        height = box.get_height()
        ax.text(box.get_x() + box.get_width() / 2, height + 0.1, label, ha='center', va='bottom')
    plt.xlim(-0.75, 6)
    figure_path = os.path.join(result_location, name+".png")
    plt.savefig(figure_path, format="png")

def main():
    """
    Execute generic classification methods on DNA methylation data
    """
    logger.info("We work on "+attribute+" classification")
    classifier_result_name = []
    classifier_result_acc = []

    classifier_result_name.append("L1")
    classifier_result_acc.append(76.36)
    classifier_result_name.append("L2")
    classifier_result_acc.append(74.55)
    visualization("LR", classifier_result_name, classifier_result_acc)
    #
    classifier_result_name.clear()
    classifier_result_acc.clear()
    classifier_result_name.append("k=2")
    classifier_result_acc.append(64.24)
    classifier_result_name.append("k=3")
    classifier_result_acc.append(68.48)
    classifier_result_name.append("k=4")
    classifier_result_acc.append(66.06)
    classifier_result_name.append("k=5")
    classifier_result_acc.append(64.85)
    visualization("KNN", classifier_result_name, classifier_result_acc)
    #
    classifier_result_name.clear()
    classifier_result_acc.clear()
    classifier_result_name.append("Linear")
    classifier_result_acc.append(76.97)
    classifier_result_name.append("RBF")
    classifier_result_acc.append(61.21)
    classifier_result_name.append("Poly 2")
    classifier_result_acc.append(61.21)
    classifier_result_name.append("Poly 3")
    classifier_result_acc.append(61.21)
    classifier_result_name.append("Poly 4")
    classifier_result_acc.append(61.21)
    visualization("SVM", classifier_result_name, classifier_result_acc)

    classifier_result_name.clear()
    classifier_result_acc.clear()
    classifier_result_name.append("Sigmoid-1")
    classifier_result_acc.append(72.12)
    classifier_result_name.append("Sigmoid-2")
    classifier_result_acc.append(73.33)
    classifier_result_name.append("Tanh-1")
    classifier_result_acc.append(76.36)
    classifier_result_name.append("Tanh-2")
    classifier_result_acc.append(76.97)
    classifier_result_name.append("ReLU-1")
    classifier_result_acc.append(70.91)
    classifier_result_name.append("ReLU-2")
    classifier_result_acc.append(76.36)
    visualization("MLP", classifier_result_name, classifier_result_acc)


if __name__ == '__main__':
    main()
