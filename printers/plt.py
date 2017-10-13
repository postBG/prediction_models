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

    plt.show()


def get_font_color(value):
    if value < 5:
        return "black"
    else:
        return "white"
