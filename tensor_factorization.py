import pandas as pd
from scipy import sparse
import numpy as np
from numpy.random import shuffle
from numpy.linalg import norm
from numpy import dot, array, zeros, setdiff1d
import random
from rescal import rescal_als
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc


def predict_rescal_als(T):
    A, R, _, _, _ = rescal_als(
        T, 45, init='nvecs', conv=1e-3 ####latent features k =45##

    )
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P


def normalize_predictions(P, e, k):
    for a in range(e):
        for b in range(e):
            nrm = norm(P[a, b, :k])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P



def create_tensor_graph(relations,entities):

    for index, row in df1.iterrows():
        subs = row["subject"].rsplit('/', 1)[-1]
        preds = row["predicate"].rsplit('/', 1)[-1]
        obj = row["object"].rsplit('/', 1)[-1]

        X[relations.index(preds)][entities.index(subs),entities.index(obj)] = 1
        #print subs, " " ,preds," ", obj,entities.index(subs),relations.index(preds),entities.index(obj)

    return X

def get_ground_truth(T):
    mat = np.zeros([e,e,k])
    for i in range(k):
        mat[:,:,i]=T[i].toarray()

    return mat


def innerfold(T, mask_idx, target_idx, e, k, sz):
    Tc = [Ti.copy() for Ti in T]
    mask_idx = np.unravel_index(mask_idx, (e, e, k))
    target_idx = np.unravel_index(target_idx, (e, e, k))

    # set values to be predicted to zero
    for i in range(len(mask_idx[0])):
        Tc[3][mask_idx[0][i], mask_idx[1][i]] = 0 ##3 is the index for predicate contains#
        #Tc[mask_idx[2][i]][mask_idx[0][i], mask_idx[1][i]] = 0

    # predict unknown values
    P = predict_rescal_als(Tc)
    P = normalize_predictions(P, e, k)
    ##Evaluation AUC-PR or AUC-ROC####
    # compute area under precision recall curve
    fpr, tpr, threshold = metrics.roc_curve(GROUND_TRUTH[target_idx],P[target_idx])
    #fpr, tpr, threshold = metrics.roc_curve(actual, predicted)
    roc_auc = metrics.auc(fpr, tpr)
    prec, recall, _ = precision_recall_curve(GROUND_TRUTH[target_idx], P[target_idx])
    #return roc_auc
    return auc(recall, prec)


df1 = pd.read_csv("Zia_data_rdf/recommendation.csv",sep = ",")

df1.columns = ["subject", "predicate", "object"]

predicate = df1["predicate"]
subs = df1["subject"]
obj = df1["object"]

preds = set()
for i in predicate:
   preds.add(i.rsplit('/', 1)[-1])

preds = list(preds)

nodes = set()

for i in subs:

   nodes.add(i.rsplit('/', 1)[-1])

for i in obj:
   nodes.add(i.rsplit('/', 1)[-1])



preds = list(preds)
nodes = list(nodes)


print preds
k =  len(preds)
e = len(nodes)

X = [sparse.lil_matrix((e,e)) for i in range(k)]
Tensors = create_tensor_graph(preds,nodes)
print('Datasize: %d x %d x %d | No. of classes: %d' % (
        Tensors[0].shape + (len(Tensors),) + (k,)))

GROUND_TRUTH = get_ground_truth(Tensors)
SZ = e * e * k

#Do cross-validation
FOLDS = 10
IDX = list(range(SZ))
shuffle(IDX)
fsz = int(SZ / FOLDS)
offset = 0
AUC_train = zeros(FOLDS)
AUC_test = zeros(FOLDS)
for f in range(FOLDS):
    idx_test = IDX[offset:offset + fsz]
    idx_train = setdiff1d(IDX, idx_test)
    shuffle(idx_train)
    idx_train = idx_train[:fsz].tolist()
    IDX1 = random.sample(xrange(SZ),fsz)
    print('Train Fold %d' % f)
    AUC_train[f] = innerfold(Tensors, idx_train + idx_test, idx_train, e, k, SZ)
    print('Test Fold %d' % f)
    AUC_test[f] = innerfold(Tensors, IDX1, IDX1, e, k, SZ)
    offset += fsz

print('AUC-PR Test Mean / Std: %f / %f' % (AUC_test.mean(), AUC_test.std()))
print('AUC-PR Train Mean / Std: %f / %f' % (AUC_train.mean(), AUC_train.std()))