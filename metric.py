from sklearn.metrics import confusion_matrix


def get_metrics(cm):
    precision = cm[1, 1] / sum(cm[:, 1])
    recall = cm[1, 1] / sum(cm[1, :])
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))

    return precision, recall, f1_score, accuracy


def get_metrics_using_labels(labels, predicted_labels):
    cm = confusion_matrix(labels, predicted_labels)
    return get_metrics(cm)
