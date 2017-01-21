import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

result_filepath = 'val_results.txt'

# the index of ky & szr are 0 and 2, respectively
is_ky = []
pred_ky = []
is_szr = []
pred_szr = []
ky_scores = []
szr_scores = []
with open(result_filepath, 'r') as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        true_label = int(tokens[1])
        pred_label = int(tokens[2])
        ky_prob = float(tokens[3])
        szr_prob = float(tokens[5])

        is_ky.append(1 if true_label == 0 else 0)
        pred_ky.append(1 if pred_label == 0 else 0)
        ky_scores.append(ky_prob)

        is_szr.append(1 if true_label == 2 else 0)
        szr_scores.append(szr_prob)

ky_cnf_mat = confusion_matrix(is_ky, pred_ky, labels=[1, 0])
print(ky_cnf_mat)

ky_fpr, ky_tpr, ky_ths = roc_curve(is_ky, ky_scores)
ky_auc = auc(ky_fpr, ky_tpr)

szr_fpr, szr_tpr, szr_ths = roc_curve(is_szr, szr_scores)
szr_auc = auc(szr_fpr, szr_tpr)

plt.plot(ky_fpr, ky_tpr, 'k--', lw=2,
         label='Kao Ya ROC curve (auc = {:.2f})'.format(ky_auc))
plt.plot(szr_fpr, szr_tpr, 'b-.', lw=2,
         label='Shui Zhu Rou ROC curve (auc = {:.2f})'.format(szr_auc))
plt.plot([0, 1], [0, 1], 'k', lw=1)
plt.plot([0, 0, 1], [0, 1, 1], 'k:', lw=2)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
