#! /usr/bin/env python

import os
import sys
import itertools
import numpy as np
from sklearn.svm import SVC


allClipFile = "allClips.txt"
labelFile = 'MEDIAEVALaffect.txt'
annotations = os.path.join(sys.argv[1],labelFile)

EVAL_DIR, allClips = sys.argv[1], map(lambda l: l[:-1], open(sys.argv[2]))
all_k = np.loadtxt(sys.stdin)

videolabels = np.genfromtxt(annotations, names=True, delimiter='\t', dtype=None)

train_labels = videolabels['valenceClass']
labels = sorted(set(train_labels))
train_labels = np.array([labels.index(x) for x in train_labels])
train_features = all_k
train_features = np.array(train_features)

REG_C = 1.0
model = SVC(kernel = 'precomputed', C = REG_C, max_iter = 10000)
model.fit(train_features, train_labels)


from sklearn.externals import joblib
joblib.dump(model, 'valencemodel.pkl')
