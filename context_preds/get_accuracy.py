from sklearn.metrics import accuracy_score
import numpy as np
import glob
preds = glob.glob("context_preds*")
machine_id_db = ['_'.join(s.split('.')[0].split('_')[3:6]) for s in preds]
gt_sorted = []
preds_sorted = []
preds_raw_sorted = []
for id in machine_id_db:
     gt = np.full(len(np.load("context_preds_expo_{}_3.npy".format(id))), int(id.split('_')[1]))
     gt_sorted.append(gt)
     preds_sorted.append(np.load("context_preds_expo_{}_3.npy".format(id)))
     preds_raw_sorted.append(np.load("oracle_preds_{}_3.npy".format(id)))
gt_c = np.concatenate(gt_sorted)
preds_c = np.concatenate(preds_sorted)
preds_raw_c = np.concatenate(preds_raw_sorted)
print("window_preds", accuracy_score(gt_c, preds_c))
print("raw preds", accuracy_score(gt_c, preds_raw_c))
