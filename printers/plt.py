import random

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def print_confusion_matrix(labels, predicted_labels, class_names=(0, 1)):
    cm = confusion_matrix(labels, predicted_labels)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap=plt.cm.Blues,
                    interpolation='nearest', vmin=0, vmax=len(labels) / 5)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=get_font_color(cm[x][y]))

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(range(width), class_names[:width], rotation=30)
    plt.yticks(range(height), class_names[:height])

    precision, recall, f1_score, accuracy = get_metrics(cm)

    plt.figtext(0.01, 0.13, "Precision: {:2f}".format(precision))
    plt.figtext(0.01, 0.09, "Recall: {:2f}".format(recall))
    plt.figtext(0.01, 0.05, "F1-score: {:2f}".format(f1_score))
    plt.figtext(0.01, 0.01, "Accuracy: {:2f}".format(accuracy))

    plt.show()


def get_metrics(cm):
    precision = cm[1, 1] / sum(cm[:, 1])
    recall = cm[1, 1] / sum(cm[1, :])
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))

    return precision, recall, f1_score, accuracy

def get_font_color(value):
    if value < 5:
        return "black"
    else:
        return "white"


if __name__ == '__main__':
    true_labels = [random.randint(1, 10) for i in range(1)]
    predicted = [random.randint(1, 10) for i in range(1)]
    print_confusion_matrix(true_labels, predicted)
