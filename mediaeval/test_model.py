#! /usr/bin/env python

import os
import sys
import itertools
import numpy as np

from sklearn.svm import SVC
from sklearn.externals import joblib

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

classifier = joblib.load('valencemodel.pkl')

test_features=train_features
test_labels = train_labels

results = classifier.predict(test_features)
num_correct = ( results == test_labels ).sum()
recall = num_correct / len(test_labels)
print "model accuracy (%): ", recall * 100 , "%"
