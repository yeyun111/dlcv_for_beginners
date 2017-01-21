from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

result_filepath = 'val_results.txt'

ky_probs = []
with open(result_filepath, 'r') as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        true_label = int(tokens[1])
        is_ky = 1 if true_label == 0 else 0
        ky_prob = float(tokens[3])
        ky_probs.append([is_ky, ky_prob])

ky_probs_sorted = np.array(sorted(ky_probs, key=itemgetter(1), reverse=True))
for is_ky, ky_prob in ky_probs_sorted:
    print('{:.0f} {:.6f}'.format(is_ky, ky_prob))

labels = ky_probs_sorted[:, 0]
probs = ky_probs_sorted[:, 1]

precision, recall, ths = precision_recall_curve(labels, probs)
ap = average_precision_score(labels, probs)

plt.figure('Kao Ya Precision-Recall Curve')
plt.plot(recall, precision, 'k', lw=2, label='Kao Ya')
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve: Average Precision={:.4f}'.format(ap))
plt.legend(loc="lower left")
plt.show()

