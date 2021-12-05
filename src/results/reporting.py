from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, f1_score, classification_report

def classification_report_2(y_true, y_pred):
    tn_00 = sum(y_pred[y_true == 0] == y_true[y_true == 0])  # true negatives
    tp_11 = sum(y_pred[y_true == 1] == y_true[y_true == 1])  # true positives
    fp_01 = sum(y_true == 0) - tn_00  # false positives
    fn_10 = sum(y_true == 1) - tp_11  # false negatives
    # confusion_matrix = np.array([[tn_00, fp_01], [fn_10, tp_11]])

    class_0_accuracy = 100.0 * sum(y_pred[y_true == 0] == y_true[y_true == 0]) / sum(y_true == 0)
    class_1_accuracy = 100.0 * sum(y_pred[y_true == 1] == y_true[y_true == 1]) / sum(y_true == 1)

    # print("Kmeans Classification Report:")
    print(f"Overall Accuracy: {round(100.0 * accuracy_score(y_true, y_pred), 2)} %")
    # print(f"F1-Score: {round(f1_score(y_true, y_pred), 3)}")
    print(f"Class 0 accuracy: {round(class_0_accuracy, 2)} %")
    print(f"Class 1 accuracy: {round(class_1_accuracy, 2)} %")

    print("Confusion Matrix:")
    confusion_matrix = PrettyTable(['', 'Predicted 0', 'Predicted 1', 'Total'])
    confusion_matrix.add_row(['Actual 0', tn_00, fp_01, tn_00 + fp_01])
    confusion_matrix.add_row(['Actual 1', fn_10, tp_11, fn_10 + tp_11])
    confusion_matrix.add_row(
        ['Total', tn_00 + fn_10, fp_01 + tp_11, tn_00 + fn_10 + fp_01 + tp_11])
    print(confusion_matrix)
    print(classification_report(y_true, y_pred))