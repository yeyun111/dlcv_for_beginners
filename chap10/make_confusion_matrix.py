import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

result_filepath = 'val_results.txt'

true_labels = []
pred_labels = []
n_correct = 0
with open(result_filepath, 'r') as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        true_label = int(tokens[1])
        pred_label = int(tokens[2])
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        n_correct += 1 if true_label == pred_label else 0

print('Accuracy = {:.2f}%'.format(float(n_correct)/float(len(true_labels))*100))
cnf_mat = confusion_matrix(true_labels, pred_labels)
foods = ['kaoya', 'yangrouchuan', 'shuizhurou', 'jitang', 'maxiao', 'miantiao', 'baozi']
plot_confusion_matrix(cnf_mat, classes=foods)
plt.show()
