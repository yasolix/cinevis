#! /usr/bin/env python

import os
import sys
import itertools
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#%matplotlib inline


#evalDir = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/Discrete/annotations/"
#allClipFile = "allClips.txt"
#labelFile = "ACCEDEranking.txt"
#annotations = os.path.join(evalDir,labelFile)

allClipFile = "allClips.txt"
labelFile = 'ACCEDEranking.txt'
annotations = os.path.join(sys.argv[1],labelFile)

EVAL_DIR, allClips = sys.argv[1], map(lambda l: l[:-1], open(sys.argv[2]))
all_k = np.loadtxt(sys.stdin)

#all_k =np.loadtxt("/home/yt/Desktop/cvpr2014/repro/mediaeval/data/kernel.txt")

videolabels = np.genfromtxt(annotations, names=True, delimiter='\t', dtype=None)

clip_id = videolabels['id'] 
name = videolabels['name']
valrank = videolabels['valenceRank']
arrank = videolabels['arousalRank']
valence = videolabels['valenceValue']
arousal = videolabels['arousalValue']
valvar = videolabels['valenceVariance']
arvar = videolabels['arousalVariance']

features = all_k
features = np.array(features)


def splitindex(x,train_perc):
    r = np.random.rand(len(x)) < train_perc
    train = x[r == 1]
    test = x[r == 0]
    return train,test

scores_rbf = list()
scores_rstd = list()
scores_lin = list()
scores_lstd = list()
scores_poly = list()
scores_pstd = list()

for i in xrange(4):
	trindex, tsindex = splitindex(clip_id,0.85)
	
	X=features[trindex]
	y=valence[trindex]

	tX=features[tsindex]
	ty=valence[tsindex]
###############################################################################
# Fit regression model
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=4)
	y_rbf = svr_rbf.fit(X, y).predict(tX)
	y_lin = svr_lin.fit(X, y).predict(tX)
	y_poly = svr_poly.fit(X, y).predict(tX)

        mse_rbf = mean_squared_error(ty, y_rbf)
        mse_lin = mean_squared_error(ty, y_lin)
        mse_poly = mean_squared_error(ty, y_poly)

	scores_rbf.append(np.mean(mse_rbf))
	scores_rstd.append(np.std(mse_rbf))

	scores_lin.append(np.mean(mse_lin))
	scores_lstd.append(np.std(mse_lin))

	scores_poly.append(np.mean(mse_poly))
	scores_pstd.append(np.std(mse_poly))

###############################################################################
# look at the results
	a="X=tsindex

	fig = plt.figure(figsize=(10, 10))
	plt.scatter(X, ty, c='k', label='data')
	plt.hold('on')
	plt.scatter(X, y_rbf, c='g', label='RBF model')
	plt.scatter(X, y_lin, c='r', label='Linear model')
	plt.scatter(X, y_poly, c='b', label='Polynomial model')

	plt.plot(X, ty, c='k', label='data')
	plt.plot(X, y_rbf, c='g', label='RBF model')
	plt.plot(X, y_lin, c='r', label='Linear model')
	plt.plot(X, y_poly, c='b', label='Polynomial model')
	plt.xlabel('data')
	plt.ylabel('target')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()
        "
print 'Mean Squared Error of rbf is %f and std is %f' % np.mean(scores_rbf) % np.mean(scores_rstd)
print 'Mean Squared Error of lin is %f and std is %f' % np.mean(scores_lin) % np.mean(scores_lstd)
print 'Mean Squared Error of poly is %f and std is %f' % np.mean(scores_poly) % np.mean(scores_pstd)
