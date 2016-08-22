#! /usr/bin/env python

import os
import sys
import itertools
import numpy as np
from sklearn.svm import SVC


#allClipFile = "allClips.txt"
labelFile = 'ACCEDEranking.txt'
annotations = os.path.join(sys.argv[1],labelFile)

EVAL_DIR, allClips = sys.argv[1], map(lambda l: l[:-1], open(sys.argv[2]))
all_k = np.loadtxt(sys.stdin)

videolabels = np.genfromtxt(annotations, names=True, delimiter='\t', dtype=None)

#train_labels = videolabels['valenceClass']
train_id = videolabels['id'] 	
train_name = videolabels['name']
train_valrank = videolabels['valenceRank']
train_arrank = videolabels['arousalRank']
train_valence = videolabels['valenceValue']
train_arousal = videolabels['arousalValue']
train_valvar = videolabels['valenceVariance']
train_arvar = videolabels['arousalVariance']


labels = sorted(set(train_labels))
train_labels = np.array([labels.index(x) for x in train_labels])
train_features = all_k
train_features = np.array(train_features)

REG_C = 1.0
vallence_model = SVC(kernel = 'precomputed', C = REG_C, max_iter = 10000)
vallence_model.fit(train_features, train_valence)

arousal_model = SVC(kernel = 'precomputed', C = REG_C, max_iter = 10000)
arousal_model.fit(train_features, train_arousal)



from sklearn.externals import joblib
joblib.dump(vallence_model, 'valencemodel.pkl')
joblib.dump(arousal_model, 'arousalmodel.pkl')
