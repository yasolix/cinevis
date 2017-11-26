
# coding: utf-8

# ## Valence-Arousal Prediction Audio and Visual Features
# 
# The Mediaeval 2017 Emotional Impact of Movies Task includes the data in the emotional domain 
# (valence - arousal  regression) and  fear ( binary classification).
# We have displayed the valence and arousal of all the movies in the dataset.
# Also the time of the movie where fear is present is specified with the value of the second.
# According to the Russell's circumplex model we were expectinf the "FEAR" to be appeared in the negative vallence, positive arousal part of the circumflex.
# However in some movies, we can see that frightment exists in positive valence with negative arousal also.
# 

# In[2]:


import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import matplotlib.colors as colors


import matplotlib
matplotlib.style.use('ggplot')

get_ipython().magic(u'matplotlib inline')

import numpy as np
import pylab as pl
import re, fileinput
import os.path
import glob
import pickle
import sys


# In[3]:

import numpy as np 
print(np.__version__) 
print(np.__path__)


# In[4]:

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import scipy
from scipy.stats import pearsonr


# In[5]:

#Dev data
#movieNames = ['After_The_Rain','Attitude_Matters','Barely_legal_stories','Between_Viewings','Big_Buck_Bunny','Chatter','Cloudland','Damaged_Kung_Fu','Decay','Elephant_s_Dream','First_Bite','Full_Service','Islands','Lesson_Learned','Norm','Nuclear_Family','On_time','Origami','Parafundit','Payload','Riding_The_Rails','Sintel','Spaceman','Superhero','Tears_of_Steel','The_room_of_franz_kafka','The_secret_number','To_Claire_From_Sonny','Wanted','You_Again']

pathcontinuous = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/ContinuousLIRIS-ACCEDE/"
continuousAnnotationsFolder = pathcontinuous +'continuous-annotations/'
devdatacontinous =  pathcontinuous + "continuous-movies/"
pathcontfeatures = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/Continuous/features-out/"

datahome = '/home/yt/Desktop/mediaeval2017'

med2017visualFeaturesfolder='/home/yt/Desktop/mediaeval2017/MEDIAEVAL17-DevSet-Visual_features/MEDIAEVAL17-DevSet-Visual_features/features/'
med2017audiofolder='/home/yt/Desktop/mediaeval2017/MEDIAEVAL17-DevSet-Audio_features/MEDIAEVAL17-DevSet-Audio_features/features/'
med2017annotationsFolder = '/home/yt/Desktop/mediaeval2017/MEDIAEVAL17-DevSet-Valence_Arousal-annotations/MEDIAEVAL17-DevSet-Valence_Arousal-annotations/annotations/'
med2017fearFolder = '/home/yt/Desktop/mediaeval2017/MEDIAEVAL17-DevSet-Fear-annotations/MEDIAEVAL17-DevSet-Fear-annotations/annotations/'
med2017dataFolder = devdatacontinous

### Test Data

med2017visualFeaturesfolderTest='/home/yt/Desktop/mediaeval2017/MEDIAEVAL17-TestSet-Visual_features/MEDIAEVAL17-TestSet-Visual_features/visual_features/'
med2017audiofolderTest = '/home/yt/Desktop/mediaeval2017/MEDIAEVAL17-TestSet-Audio_features/MEDIAEVAL17-TestSet-Audio_features/audio_features/'
med2017datafolderTest = '/home/yt/Desktop/mediaeval2017/MEDIAEVAL17-TestSet-Data/MEDIAEVAL17-TestSet-Data/data/'

med2017testfeatures = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/Continuous/features-out/"


# In[ ]:




# In[ ]:




# In[9]:

files = glob.glob(med2017datafolderTest+'*')
testmovieNames =[ f.split('/')[-1].replace('.mp4','') for f in sorted(files) ]


# In[10]:

files = glob.glob(med2017dataFolder+'*')
movieNames =[ f.split('/')[-1].replace('.mp4','') for f in sorted(files) ]


# In[11]:

movieNames,testmovieNames


# In[12]:

fpsMovie = [['After_The_Rain',23.976],
            ['Attitude_Matters',29.97],
            ['Barely_legal_stories',23.976],
            ['Between_Viewings',25],
            ['Big_Buck_Bunny',24],
            ['Chatter',24],
                ['Cloudland',25],
                ['Damaged_Kung_Fu',25],
                ['Decay',23.976],
                ['Elephant_s_Dream',24],
                ['First_Bite',25],
                ['Full_Service',29.97],
                ['Islands',23.976],
                ['Lesson_Learned',29.97],
                ['Norm',25],
                ['Nuclear_Family',23.976],
                ['On_time',30],
                ['Origami',24],
                ['Parafundit',24],
                ['Payload',25],
                ['Riding_The_Rails',23.976],
                ['Sintel',24],
                ['Spaceman',23.976],
                ['Superhero',29.97],
                ['Tears_of_Steel',24],
                ['The_room_of_franz_kafka',29.786],
                ['The_secret_number',23.976],
                ['To_Claire_From_Sonny',23.976],
                ['Wanted',25],
                ['You_Again',29.97]]

contmoviesfps = pd.DataFrame(fpsMovie,columns=['name','fps'])
#contmoviesfps.set_index('name', inplace=True)
#contmoviesfps.index.name = None
#contmoviesfps['After_The_Rain']


# In[13]:

contmoviesfps.plot.line()


# In[41]:

contmoviesfps['f'] = np.round(contmoviesfps['fps'])
contmoviesfps


# In[43]:

def getfps(movname):
    return contmoviesfps[ contmoviesfps.name == movname ]['f']


# In[44]:

print contmoviesfps[ contmoviesfps.name == 'You_Again' ]['f']
print getfps('You_Again')


# In[18]:

movgroups_wodecay = {
    0:['You_Again','Damaged_Kung_Fu','The_secret_number','Spaceman'],
    1:['Cloudland','Origami','Riding_The_Rails','Tears_of_Steel','Sintel'],
    2:['On_time','Elephant_s_Dream','Norm','Big_Buck_Bunny','Chatter','Full_Service'],
    3:['Islands','To_Claire_From_Sonny','Nuclear_Family','After_The_Rain','Parafundit'],
    4:['The_room_of_franz_kafka','Attitude_Matters','Lesson_Learned','Superhero'],
    5:['First_Bite','Wanted','Between_Viewings','Barely_legal_stories','Payload']
}

movgroups = {
    0:['You_Again','Damaged_Kung_Fu','The_secret_number','Spaceman'],
    1:['Cloudland','Origami','Riding_The_Rails','Tears_of_Steel','Sintel'],
    2:['On_time','Elephant_s_Dream','Norm','Big_Buck_Bunny','Chatter','Full_Service'],
    3:['Islands','To_Claire_From_Sonny','Nuclear_Family','After_The_Rain','Parafundit'],
    4:['The_room_of_franz_kafka','Attitude_Matters','Lesson_Learned','Superhero'],
    5:['First_Bite','Wanted','Between_Viewings','Barely_legal_stories','Payload'],
    6:['Decay']
}

mov2groups = {
    0:['Decay'],
    1:['You_Again','Damaged_Kung_Fu','The_secret_number','Spaceman'],
    2:['Cloudland','Origami','Riding_The_Rails','Tears_of_Steel','Sintel'],
    3:['On_time','Elephant_s_Dream','Norm','Big_Buck_Bunny','Chatter','Full_Service'],
    4:['Islands','To_Claire_From_Sonny','Nuclear_Family','After_The_Rain','Parafundit'],
    5:['The_room_of_franz_kafka','Attitude_Matters','Lesson_Learned','Superhero'],
    6:['First_Bite','Wanted','Between_Viewings','Barely_legal_stories','Payload'],
}


def gettraintestmovielist(mlist,groups=movgroups):
    testlist = groups[mlist]
    trainlist =[]
    for idx, group in enumerate(groups):
        if idx != mlist:
            for g in groups[idx]:
                trainlist.append(g)
    return trainlist, testlist

def gettraintest2movielist(foldno,groups=mov2groups):
    if foldno==1:
        mlist=[1,2]
    elif foldno==2:
        mlist=[3,4]
    elif foldno==3:
        mlist=[5,6]
    elif foldno==4:
        mlist=[2,3]
    elif foldno==5:
        mlist=[4,5]
    else:
        mlist=[]
    
    testlist = [] 
    for i in mlist:
        for f in groups[i]:
            testlist.append(f)
            
    trainlist =[]
    for idx, group in enumerate(groups):
        for f in groups[idx]:
            if f not in testlist:
                trainlist.append(f)
                
    return trainlist, testlist


# In[19]:

gettraintest2movielist(4)


# ## Valence - Arosal Annotations
# Thank you for downloading LIRIS-ACCEDE dataset.
# This file contains valence/arousal annotations for the LIRIS-ACCEDE continuous part that is used for the first subtask of the MEDIAEVAL 2017 Emotional Impact of Movies task.
# For each of the 30 movies, consecutive ten seconds-segments sliding over the whole movie with a shift of 5 seconds are considered and provided with valence and arousal annotations.
# Each txt file contains 4 columns separated by tabulations. The first column is the segment id, starting from 0, the second column is the starting time of the segment in the movie and the third and fourth columns are respectively the valence and arousal values for this segment.

# In[20]:

def getAnnotationDf(movname,folder=med2017annotationsFolder):
    filename = os.path.join(folder, movname + '-MEDIAEVAL2017-valence_arousal.txt')
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    return df


# In[22]:

df = getAnnotationDf(movieNames[0])
df.hist(alpha=0.5,bins=50)


# In[23]:

#df.head()
#df.describe()


# ## Valence, Arousal histogram plots for Dev-Set

# In[24]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(movieNames):
    if (ii+1 > 30):
        plt.subplot(6,5,ii)
    else :
        plt.subplot(6,5,ii+1)
    df = getAnnotationDf(mov)
    df[['MeanValence','MeanArousal']].plot.hist(ax=plt.gca(),title=mov,alpha=0.5,bins=50)


# ## Valence , Arousal plots for Dev-Set

# In[25]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    df = getAnnotationDf(mov)
    df[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)
    #.hist(alpha=0.5,bins=50)


# ## Fear Annotations

# In[26]:

def getFearDf(movname):
    filename = os.path.join(med2017fearFolder, movname + '-MEDIAEVAL2017-fear.txt')
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    return df


# In[27]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    df = getFearDf(mov)
    df[['Fear']].plot(ax=plt.gca(),title=mov)


# ## Audio Features
# 

# In[28]:

def getAudioDf(moviename,folder=med2017audiofolder):
    if 'TestSet' in folder:
        files = glob.glob(folder+moviename+'/audio_features/*.csv')
    else:
        files = glob.glob(folder+moviename+'/*.csv')
    files = sorted(files)
    files
    alist = []
    for fname in files:
        f=open(fname,'r')
        h = []
        for l in f :
            if '@attribute' in l:
                h.append(l.split()[1])
            elif l == '\n':
                l
            elif l[0] =='@':
                l
            else:
                alist.append(map(float,l.split(',')[1:])) #first attribute is string ,skipped
        f.close()
    
    return pd.DataFrame(alist,columns=h[1:])


# ## Visual Features

# In[29]:

visual_feat = ['acc', 'cedd', 'cl', 'eh', 'fc6', 'fcth', 
               'gabor', 'jcd', 'lbp', 'sc', 'tamura'   ]
visual_feat_wofc16 = ['acc', 'cedd', 'cl', 'eh', 'fcth', 
               'gabor', 'jcd', 'lbp', 'sc', 'tamura'   ]


# In[30]:

def getVisFeatureDf(moviename,typename,folder=med2017visualFeaturesfolder):
    files = glob.glob(folder+moviename+'/'+typename+'/*.txt')
    files = sorted(files)
    alist = []
    for fname in files:
        f=open(fname,'r')
        for l in f:
            alist.append(map(float,l.split(',')))
        f.close()
    return pd.DataFrame(alist)

def getAvgVisFeatureDf(moviename,typename,folder=med2017visualFeaturesfolder):
    df = getVisFeatureDf(moviename,typename,folder)
    dfwindow = df.rolling(10).mean()[9::5] ############### start with 9
    dfwindow.reset_index(inplace=True)
    dfwindow.drop('index',axis=1,inplace=True)
    return dfwindow

def getAvgVisFeatListDf(moviename,featlist,folder=med2017visualFeaturesfolder):
    df = getVisFeatureDf(moviename,featlist[0],folder)
    for feat in featlist[1:]:
        tdf = getVisFeatureDf(moviename,feat,folder)
        df = pd.concat([df,tdf],axis=1)
    
    dfwindow = df.rolling(10).mean()[9::5] ############### start with 9
    dfwindow.reset_index(inplace=True)
    dfwindow.drop('index',axis=1,inplace=True)
    dfwindow.columns=list(range(len(dfwindow.columns)))
    return dfwindow


# In[31]:

sum([len(getAnnotationDf(m)) for m in movieNames ])


# In[30]:

sum([len(getAudioDf(m)) for m in movieNames ])


# In[31]:

sum([len(getVisFeatureDf(m,'cl')) for m in movieNames ])


# In[32]:

sum([len(getAvgVisFeatureDf(m,'cl')) for m in movieNames ])


# In[61]:

df = getVisFeatureDf(movieNames[0],'cl')
#df = getAvgVisFeatureDf(movieNames[0],'cl')
#df = getAvgVisFeatListDf(movieNames[0],['cl','eh'])
#df.hist()
df.head(10)


# In[18]:

#df = getAvgVisFeatListDf(movieNames[0],['fc6'])
#df = getVisFeatureDf(movieNames[0],'fc6')
#df.describe()


# ## Low Level Cinematographic Features
# fps değerlerine göre, feature çıkarma key frame seçme ve averaging tekrar yapılacak.

# In[81]:

def getLowFeatureDf(movname):
    fname = movname +'.mp4continous_features.txt'
    df = pd.DataFrame(np.genfromtxt( os.path.join(pathcontfeatures,fname)))
    df.columns = ['time','framemean','huemean','satmean','valmean', 'redmean','greenmean','bluemean', 'lummean','motion']
    return df

def getLowFeature10SecDf(movname):
    pdf = getLowFeatureDf(movname)
    fps = getfps(movname)
    dfwindow = pdf.rolling(10).mean()[9::5]
    dfwindow.reset_index(inplace=True)
    dfwindow.drop('index',axis=1,inplace=True)
    dfwindow.drop('time',axis=1,inplace=True)
    return dfwindow

def getMovieListLowFeatFearDf(movieNames):
    X = getLowFeature10SecDf(movieNames[0])
    y = getFearDf(movieNames[0]).Fear[:len(X)]

    for mov in movieNames[1:]:
        tX=getLowFeatureDf(mov)
        ty=getFearDf(mov).Fear[:len(tX)]
        X = X.append(tX)
        y = y.append(ty)
        if (X.shape != y.shape):
            print mov, X.shape, y.shape
    return X,y


# In[70]:

#getLowFeatureDf(movieNames[1]).head(10)[2::2]


# In[71]:

#print getLowFeatureDf(movieNames[1]).head(10)
#print getLowFeatureDf(movieNames[1]).head(10).mean()
print getLowFeature10SecDf(movieNames[1]).head(10)


# ## Train and Test set creation

# In[85]:

def getFeatureswFearDf(movieNames,featlist=visual_feat_wofc16):
    Xv = getAvgVisFeatListDf(movieNames[0],featlist)
    Xa = getAudioDf(movieNames[0])
    Xd = getAvgVisFeatListDf(movieNames[0],['fc6'])
    Xl = getLowFeature10SecDf(movieNames[0])
    y = getFearDf(movieNames[0])[['Fear']]
    
    mlen = min(len(Xv),len(Xa), len(Xd), len(Xl),len(y))
    
    Xv = Xv[:mlen]
    Xa = Xa[:mlen]
    Xd = Xd[:mlen]
    Xl = Xl[:mlen]
    y  = y[:mlen]
    
    for mov in movieNames[1:]:
        tXv = getAvgVisFeatListDf(mov,featlist)
        tXa = getAudioDf(mov)
        tXd = getAvgVisFeatListDf(mov,['fc6'])
        tXl = getLowFeature10SecDf(mov)
        ty = getFearDf(mov)[['Fear']]
        
        mlen = min(len(tXv),len(tXa),len(tXd),len(ty))
        tXv = tXv[:mlen]
        tXa = tXa[:mlen]
        tXd = tXd[:mlen]
        tXl = tXl[:mlen]
        ty = ty[:mlen]
        
        Xv  = Xv.append(tXv)
        Xa  = Xa.append(tXa)
        Xd = Xd.append(tXd)
        Xl = Xl.append(tXl)
        y  = y.append(ty)
        
    return Xv,Xa,Xd,Xl,y


# In[77]:

def getFeatureswAnnotationsDf(movieNames,featlist=visual_feat_wofc16):
    Xv = getAvgVisFeatListDf(movieNames[0],featlist)
    Xa = getAudioDf(movieNames[0])
    Xd = getAvgVisFeatListDf(movieNames[0],['fc6'])
    Xl = getLowFeature10SecDf(movieNames[0])
    y = getAnnotationDf(movieNames[0])[['MeanValence','MeanArousal']]
    
    mlen = min(len(Xv),len(Xa), len(Xd), len(Xl),len(y))
    
    Xv = Xv[:mlen]
    Xa = Xa[:mlen]
    Xd = Xd[:mlen]
    Xl = Xl[:mlen]
    y = y[:mlen]
    
    for mov in movieNames[1:]:
        tXv = getAvgVisFeatListDf(mov,featlist)
        tXa = getAudioDf(mov)
        tXd = getAvgVisFeatListDf(mov,['fc6'])
        tXl = getLowFeature10SecDf(mov)
        ty = getAnnotationDf(mov)[['MeanValence','MeanArousal']]
        
        mlen = min(len(tXv),len(tXa),len(tXd),len(ty))
        tXv = tXv[:mlen]
        tXa = tXa[:mlen]
        tXd = tXd[:mlen]
        tXl = tXl[:mlen]
        ty = ty[:mlen]
        
        Xv  = Xv.append(tXv)
        Xa  = Xa.append(tXa)
        Xd = Xd.append(tXd)
        Xl = Xl.append(tXl)
        y  = y.append(ty)
        
    return Xv,Xa,Xd,Xl,y


# In[73]:

def getMovListAudioVisFeatListwAnnotationsDf(movieNames,featlist):
    Xv = getAvgVisFeatureDf(movieNames[0],featlist[0])
    Xa = getAudioDf(movieNames[0])
    y = getAnnotationDf(movieNames[0])[['MeanValence','MeanArousal']]
    
    mlen = min(len(Xv),len(Xa),len(y))
    print(mlen)
    
    Xv = Xv[:mlen]
    Xa = Xa[:mlen]
    y = y[:mlen]
    
    for feattype in featlist[1:]:
        fXv = getAvgVisFeatureDf(movieNames[0],feattype)[:mlen]
        Xv = pd.concat( [Xv,fXv], axis=1 )

    for mov in movieNames[1:]:
        tXv = getAvgVisFeatureDf(mov,featlist[0])
        tXa = getAudioDf(mov)
        ty = getAnnotationDf(mov)[['MeanValence','MeanArousal']]
        
        mlen = min(len(tXv),len(tXa),len(ty))
        print(mlen)
        
        tXv = tXv[:mlen]
        tXa = tXa[:mlen]
        ty = ty[:mlen]
        
        for feattype in featlist[1:]:
            fXv = getAvgVisFeatureDf(mov,feattype)[:mlen]
            tXv = pd.concat( [tXv,fXv], axis=1 )
        
        Xv  = Xv.append(tXv)
        Xa  = Xa.append(tXa)
        y  = y.append(ty)
        
    return Xv,Xa,y


# In[74]:

def df2mat(df):
    return df.as_matrix().reshape((len(df),))


# # Classification work

# In[75]:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#from sklearn import cross_validation
from sklearn import metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams

def getGridCV(pipe,paramgirid,Xtrain,ytrain): # scoring ?
    grid = GridSearchCV(pipe, param_grid, cv=5,n_jobs=125)
    grid.fit(Xtrain,ytrain)
    
    return grid



# In[76]:

def modelfit(alg, X, y, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(X, y)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, X, y, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, X.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


# In[ ]:

trainlist, testlist=gettraintest2movielist(1,mov2groups)  # index 1 olanları test , diğerlerini train yapan fonksiyon
tXv,tXa,tXd,ty = getFeatureswFearDf(trainlist)
print(tXv.shape,tXa.shape,tXd.shape,ty.shape)
testXv, testXa, testXd, testy = getFeatureswFearDf(testlist)
print(testXv.shape, testXa.shape,testXd.shape, testy.shape)


# In[ ]:




# In[ ]:

param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}

pipegrad = GradientBoostingClassifier(learning_rate=0.05, 
                           n_estimators=60,max_depth=9,
                           max_features='sqrt', subsample=0.8, 
                           random_state=10) 

gsearch3 = GridSearchCV(pipegrad , param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch3.fit(Xtraina,ytrain)
#gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[ ]:


pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler()],
     'classifier__gamma': [0.001, 0.01, 1, 10],
     'classifier__C': [0.01, 1, 10,100]},
    {'classifier': [RandomForestClassifier],
     'preprocessing': [None],
     'classifier__n_estimators': [50,100,300]
     'classifier__max_features': [3,5,10]}]

grid = GridSearchCV(pipe, param_grid, cv=5)
#grid.fit(Xtraina,ytrain)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
#grid.grid_scores_, grid.best_params_, grid.best_score_


# In[ ]:

from joblib import Parallel, delayed

import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...

def trainPipe(ii,pipe,valorar, modality):
    
    #rows = []
    trainlist, testlist=gettraintest2movielist(ii,mov2groups)  # index 1 olanları test , diğerlerini train yapan fonksiyon
    tXv,tXa,tXd,ty = getFeatureswAnnotationsDf(trainlist)
    print(tXv.shape,tXa.shape,tXd.shape,ty.shape)
    testXv, testXa, testXd, testy = getFeatureswAnnotationsDf(testlist)
    print(testXv.shape, testXa.shape,testXd.shape, testy.shape)

    if modality == 'visual':
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXv,ty[[valorar]], 
                                           testXv, testy[[valorar]])
    else:
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXa,ty[[valorar]], 
                                           testXa, testy[[valorar]])
        
    return ii,mse,prs,p1

def processGroup(ii,pipe,valorar, modality):
    
    #rows = []
    trainlist, testlist=gettraintest2movielist(ii,mov2groups)  # index 1 olanları test , diğerlerini train yapan fonksiyon
    tXv,tXa,tXd,ty = getFeatureswAnnotationsDf(trainlist)
    print(tXv.shape,tXa.shape,tXd.shape,ty.shape)
    testXv, testXa, testXd, testy = getFeatureswAnnotationsDf(testlist)
    print(testXv.shape, testXa.shape,testXd.shape, testy.shape)

    if modality == 'visual':
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXv,ty[[valorar]], 
                                           testXv, testy[[valorar]])
    else:
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXa,ty[[valorar]], 
                                           testXa, testy[[valorar]])
        
    return [ii,mse,prs]

def crossgroups(pipe,valorar,modality):
    #inputs=range(len(movgroups))
    inputs=[1, 2, 3, 4, 5]
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(processGroup)(i,pipe,valorar,modality) for i in inputs)
    pipescores = pd.DataFrame(results,columns=['test-group','MSE','PCC'])

    return pipescores


# In[ ]:




# # Regression work

# In[ ]:

#trainlist, testlist=gettraintestmovielist(2,movgroups_wodecay)  # index 1 olanları test , diğerlerini train yapan fonksiyon
#trainlist, testlist
#for ii in range(len(movgroups)):
#    trnlist, tstlist=gettraintestmovielist(ii)


# In[92]:

get_ipython().run_cell_magic(u'time', u'', u'allXv,allXa,allXd,ally = getFeatureswAnnotationsDf(movieNames)\nprint(allXv.shape,allXa.shape,allXd.shape,ally.shape)')


# In[78]:

trainlist, testlist=gettraintest2movielist(2)  # index 1 olanları test , diğerlerini train yapan fonksiyon

tXv,tXa,tXd,tXl,ty = getFeatureswAnnotationsDf(trainlist)
print(tXv.shape,tXa.shape,tXd.shape,tXl.shape,ty.shape)
testXv, testXa, testXd, testXl, testy = getFeatureswAnnotationsDf(testlist)
print(testXv.shape, testXa.shape,testXd.shape,testXd.shape, testy.shape)

X_train, X_test, y_train, y_test = train_test_split(tXv, ty,test_size=0.2, random_state=0)
Xa_train, Xa_test, ya_train, ya_test = train_test_split(tXa, ty,test_size=0.2, random_state=0)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(tXd, ty,test_size=0.2, random_state=0)


# In[88]:

Xl_train, Xl_test, yl_train, yl_test = train_test_split(tXl, ty,test_size=0.2, random_state=0)
Xl_train.shape, Xl_test.shape, yl_train.shape, yl_test.shape


# ## Linear Regression - Valence

# In[89]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn import  linear_model\nfrom sklearn.metrics import mean_squared_error, r2_score\n\n# Create linear regression object\nvisual_regr = linear_model.LinearRegression()\naudio_regr = linear_model.LinearRegression()\nnn_regr = linear_model.LinearRegression()\nllf_regr = linear_model.LinearRegression() #low level features\n\n# Train the model using the training sets\nvisual_regr.fit(X_train, y_train[['MeanValence']].as_matrix().reshape((len(y_train))))\naudio_regr.fit(Xa_train, ya_train[['MeanValence']].as_matrix().reshape((len(ya_train))))\nnn_regr.fit(Xd_train, yd_train[['MeanValence']].as_matrix().reshape((len(yd_train))))\nllf_regr.fit(Xl_train, yl_train[['MeanValence']].as_matrix().reshape((len(yl_train))))\n\n# Make predictions using the testing set\nvisual_y_pred = visual_regr.predict(X_test)\naudio_y_pred = audio_regr.predict(Xa_test)\nnn_y_pred = nn_regr.predict(Xd_test)\nllf_y_pred = llf_regr.predict(Xl_test)\n")


# In[90]:

# The coefficients
print('Visual Coefficients: \n', visual_regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df2mat(y_test[['MeanValence']]), visual_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df2mat(y_test[['MeanValence']]),visual_y_pred))

print('pearson score  ',pearsonr(df2mat(y_test[['MeanValence']]),visual_y_pred))
print
print('Audio Coefficients: \n', audio_regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df2mat(ya_test[['MeanValence']]), audio_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df2mat(ya_test[['MeanValence']]),audio_y_pred))
print('pearson score  ',pearsonr(df2mat(ya_test[['MeanValence']]),audio_y_pred))
print
print('FC16 Coefficients: \n', nn_regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df2mat(yd_test[['MeanValence']]), nn_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df2mat(yd_test[['MeanValence']]),nn_y_pred))

print('pearson score  ',pearsonr(df2mat(yd_test[['MeanValence']]),nn_y_pred))

print('Low Level Cinematographic: \n', llf_regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df2mat(yl_test[['MeanValence']]), llf_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df2mat(yl_test[['MeanValence']]),llf_y_pred))

print('pearson score  ',pearsonr(df2mat(yl_test[['MeanValence']]),llf_y_pred))


# ## Grid Search on Visual Features- Valence

# In[91]:

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def getGridCV(pipe,paramgirid,Xtrain,ytrain,jobs=4): # scoring ? # jobs --> number of cores
    grid = GridSearchCV(pipe, param_grid, cv=5,n_jobs=jobs)  
    grid.fit(Xtrain,ytrain)
    
    return grid

    


# In[98]:

get_ipython().run_cell_magic(u'time', u'', u'\n#X_train, X_test, y_train, y_test \n#pipe = Pipeline([(\'preprocessing\', StandardScaler()),(\'reduce_dim\', PCA()) ,(\'classifier\', SVR())])\npipe = Pipeline([(\'preprocessing\', StandardScaler()), (\'classifier\', SVR())])\n\nparam_grid = [\n    {\'classifier\': [SVR()], \n     \'preprocessing\': [StandardScaler()],\n#     \'reduce_dim\': [PCA()],\n#     \'reduce_dim__n_components\' : [ 800],\n     \'classifier__gamma\': [0.0001, 0.001,0.01, 0.1, 1, 10, 100],\n     \'classifier__C\': [0.001, 0.01, 0.1, 1, 10, 100,200]},\n    {\'classifier\': [RandomForestRegressor(n_estimators=100)],\n     \'preprocessing\': [None], \'classifier__max_features\': [3,5,10]}]\n\ngrid_vis_valence = getGridCV(pipe,param_grid,X_train,df2mat(y_train[[\'MeanValence\']]))\n\nprint("Best params:\\n{}\\n".format(grid_vis_valence.best_params_))\nprint("Best cross-validation score: {:.2f}".format(grid_vis_valence.best_score_))\nprint("All grid scores")\n\ngrid_vis_valence.grid_scores_, grid_vis_valence.best_params_, grid_vis_valence.best_score_')


# In[99]:

tXv.shape,tXa.shape,ty.shape,
testXv.shape, testXa.shape, testy.shape


# In[100]:

#scores[0]
#scores.to_csv('grid_vis_valence.txt')


# In[101]:

def gridscores(grid):
    scores = grid.grid_scores_
    rows = []
    params = sorted(scores[0].parameters)
    for row in scores:
        mean = row.mean_validation_score
        std = row.cv_validation_scores.std()
        rows.append([mean, std] + [row.parameters['classifier']])
    scores = pd.DataFrame(rows, columns=['mean_', 'std_'] + ['classifier'])
    #scores.to_csv(filename)
    return scores


# In[105]:

gridscores(grid_vis_valence).tail()


# ## Grid Search on Low Level Cinematographic Features- Valence

# In[93]:

get_ipython().run_cell_magic(u'time', u'', u'\n#Xl_train, Xl_test, yl_train, yl_test \n#pipe = Pipeline([(\'preprocessing\', StandardScaler()),(\'reduce_dim\', PCA()) ,(\'classifier\', SVR())])\npipe = Pipeline([(\'preprocessing\', StandardScaler()), (\'classifier\', SVR())])\n\nparam_grid = [\n    {\'classifier\': [SVR()], \n     \'preprocessing\': [StandardScaler()],\n#     \'reduce_dim\': [PCA()],\n#     \'reduce_dim__n_components\' : [ 800],\n     \'classifier__gamma\': [0.0001, 0.001,0.01, 0.1, 1, 10, 100],\n     \'classifier__C\': [0.001, 0.01, 0.1, 1, 10, 100,200]},\n    {\'classifier\': [RandomForestRegressor(n_estimators=20)],\n     \'preprocessing\': [None], \'classifier__max_features\': [3,5,9]}]\n\ngrid_llf_valence = getGridCV(pipe,param_grid,Xl_train,df2mat(yl_train[[\'MeanValence\']]))\n\nprint("Best params:\\n{}\\n".format(grid_llf_valence.best_params_))\nprint("Best cross-validation score: {:.2f}".format(grid_llf_valence.best_score_))\nprint("All grid scores")\n\ngrid_llf_valence.grid_scores_, grid_llf_valence.best_params_, grid_llf_valence.best_score_')


# In[104]:

gridscores(grid_llf_valence).tail()


# ## Metrics and Paralell crossvalidation

# In[111]:

from sklearn import metrics
from scipy.stats import pearsonr

def getMetrics(y,y_pred):
    # calculate MAE using scikit-learn
    #mae = metrics.mean_absolute_error(ytestarray, y_pred)
    #print("MAE score: {:.5f}".format(mae))
    
    mse = metrics.mean_squared_error(y, y_pred)
    # calculate MSE using scikit-learn
    print("MSE score: {:.5f}".format(mse))

    # calculate RMSE using scikit-learn
    #print("RMSE: {:.5f}".format(np.sqrt(metrics.mean_squared_error(ytestarray, y_pred))))

    print("Pearson score:")
    prs = pearsonr(y,y_pred)
    print(prs)
    
    return mse,prs


# In[112]:

def evaluate_pipe(pipe,trainX,trainy,testX,testy):
    
    ytrainarray = trainy.as_matrix().reshape((len(trainy),))
    ytestarray = testy.as_matrix().reshape((len(testy),))

    pipe.fit(trainX, ytrainarray)
    
    print("Train score: {:.2f}".format(pipe.score(trainX, ytrainarray)))
    print("Test score: {:.2f}".format(pipe.score(testX, ytestarray)))

    y_pred = pipe.predict(testX)

    mse, prs = getMetrics(ytestarray,y_pred)
    
    return y_pred,mse,prs[0],pipe


# In[113]:

from joblib import Parallel, delayed

import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...

def trainPipe(ii,pipe,valorar, modality):
    
    #rows = []
    trainlist, testlist=gettraintest2movielist(ii,mov2groups)  # index 1 olanları test , diğerlerini train yapan fonksiyon
    tXv,tXa,tXd,tXl,ty = getFeatureswAnnotationsDf(trainlist)
    #print(tXv.shape,tXa.shape,tXd.shape, tXl.shape,ty.shape)
    
    testXv, testXa, testXd, testXl, testy = getFeatureswAnnotationsDf(testlist)
    #print(testXv.shape, testXa.shape,testXd.shape, testXl.shape, testy.shape)

    if modality == 'visual':
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXv,ty[[valorar]], 
                                           testXv, testy[[valorar]])
    elif modality == 'audio':
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXa,ty[[valorar]], 
                                           testXa, testy[[valorar]])
    elif modality == 'deep':
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXd,ty[[valorar]], 
                                           testXd, testy[[valorar]])    
    else: ## lllf low level features
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXl,ty[[valorar]], 
                                           testXl, testy[[valorar]])
        
        
    return ii,mse,prs,p1

def processGroup(ii,pipe,valorar, modality):
    
    #rows = []
    trainlist, testlist=gettraintest2movielist(ii,mov2groups)  # index 1 olanları test , diğerlerini train yapan fonksiyon
    tXv,tXa,tXd,tXl,ty = getFeatureswAnnotationsDf(trainlist)
    #print(tXv.shape,tXa.shape,tXd.shape, tXl.shape,ty.shape)
    
    testXv, testXa, testXd, testXl, testy = getFeatureswAnnotationsDf(testlist)
    #print(testXv.shape, testXa.shape,testXd.shape, testXl.shape, testy.shape)

    if modality == 'visual':
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXv,ty[[valorar]], 
                                           testXv, testy[[valorar]])
    elif modality == 'audio':
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXa,ty[[valorar]], 
                                           testXa, testy[[valorar]])
    elif modality == 'deep':
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXd,ty[[valorar]], 
                                           testXd, testy[[valorar]])    
    else: ## lllf low level features
        y_pred_test,mse,prs,p1 = evaluate_pipe(pipe, 
                                           tXl,ty[[valorar]], 
                                           testXl, testy[[valorar]])
    return [ii,mse,prs]

def crossgroups(pipe,valorar,modality):
    #inputs=range(len(movgroups))
    inputs=[1, 2, 3, 4, 5]
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(processGroup)(i,pipe,valorar,modality) for i in inputs)
    pipescores = pd.DataFrame(results,columns=['test-group','MSE','PCC'])

    return pipescores


# In[114]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_visual_valence = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))")


# In[115]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_visual_arousal = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n")


# In[116]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_audio_valence = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n)")


# In[117]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_audio_arousal = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    #PCA(n_components=800),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n)")


# In[118]:

ii1,mse1,prs1,pva = trainPipe(4,pipe_visual_valence,'MeanValence','visual')
ii2,mse2,prs2,pvv = trainPipe(3,pipe_audio_valence,'MeanValence','audio')
ii3,mse3,prs3,pav = trainPipe(1,pipe_visual_arousal,'MeanArousal','visual')
ii4,mse4,prs4,paa = trainPipe(1,pipe_audio_arousal,'MeanArousal','audio')


# In[55]:

# cross validation takes too much time
# do not run if it not necessary
paa_scores = crossgroups(pipe_audio_arousal,'MeanArousal','audio')
pva_scores = crossgroups(pipe_audio_valence,'MeanValence','audio')
pav_scores = crossgroups(pipe_visual_arousal,'MeanArousal','visual')
pvv_scores = crossgroups(pipe_visual_valence,'MeanValence','visual')

paa_scores.sort_values('PCC', ascending=False)
pav_scores.sort_values('PCC', ascending=False)
pva_scores.sort_values('PCC', ascending=False)
pvv_scores.sort_values('PCC', ascending=False)


# ## Smoothing

# In[119]:

def getAVprediction(f):
    audiodf = getAudioDf(f)
    visualdf = getAvgVisFeatListDf(f,visual_feat_list)
    annotdf = getAnnotationDf(f)
    ya = df2mat(annotdf[['MeanArousal']])
    yv = df2mat(annotdf[['MeanValence']])
        
    print(audiodf.shape,visualdf.shape,annotdf.shape)

    mlen = min(len(audiodf),len(visualdf))

    audiodf = audiodf[:mlen]
    visualdf = visualdf[:mlen]
    ya = ya[:mlen]
    yv = yv[:mlen]

    aa = paa.predict(audiodf)
    av = pav.predict(visualdf)
        
    va = pvv.predict(audiodf)
    vv = pva.predict(visualdf)

    df =pd.DataFrame(np.transpose([vv, va , aa, av ]), columns=['MeanValenceAudio','MeanValenceVisual','MeanArousalAudio','MeanArousalVisual'])

    return df


# In[120]:

import numpy as np
 
def holt_winters_second_order_ewma( x, span, beta ):
    N = x.size
    alpha = 2.0 / ( 1 + span )
    s = np.zeros(( N, ))
    b = np.zeros(( N, ))
    s[0] = x[0]
    for i in range( 1, N ):
        s[i] = alpha * x[i] + ( 1 - alpha )*( s[i-1] + b[i-1] )
        b[i] = beta * ( s[i] - s[i-1] ) + ( 1 - beta ) * b[i-1]
    return s
 


# In[123]:

mov =movieNames[0]
aa = getAVprediction(mov)
dfa = getAnnotationDf(mov)
smooth10 = holt_winters_second_order_ewma( df2mat(aa[['MeanValenceAudio']]), 10, 0.3 )
smooth5 = holt_winters_second_order_ewma( df2mat(aa[['MeanValenceAudio']]), 5, 0.3 )
smooth2 = aa[['MeanValenceAudio']].rolling(window=10).mean()
smooth20 = holt_winters_second_order_ewma( df2mat(aa[['MeanValenceAudio']]), 20, 0.3 )

dfa[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)
aa[['MeanValenceAudio']].plot(ax=plt.gca(), style=['*-'], title=mov)
#pd.DataFrame(smooth10,columns=['smooth10']).plot(ax=plt.gca(),style=['.-'],title=mov)
#pd.DataFrame(smooth5,columns=['smooth5']).plot(ax=plt.gca(),style=['g+-'],title=mov)
pd.DataFrame(smooth20,columns=['smooth20']).plot(ax=plt.gca(),style=['yx-'],title=mov)
#ax=plt.gca()
#plt.plot(smooth1,label='smoothing')
#plt.plot(smooth2)


# In[125]:

smooth10.shape,aa.shape


# ## Generating N-fold csv

# In[126]:

visual_feat_list= ['acc', 'cedd', 'cl', 'eh', 'fcth', 
               'gabor', 'jcd', 'lbp', 'sc', 'tamura'   ]


# In[128]:

get_ipython().run_cell_magic(u'time', u'', u"# Visual\npipe_visual_valence = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\npipe_visual_arousal = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\n# Audio\npipe_audio_valence = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))\n\npipe_audio_arousal = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    #PCA(n_components=800),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))\n\n# FC16 -->deep fetures\npipe_deep_valence = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\npipe_deep_arousal = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\n# Low Level Features\npipe_llf_valence = make_pipeline(\nRandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,\n           max_features=9, max_leaf_nodes=None, min_impurity_split=1e-07,\n           min_samples_leaf=1, min_samples_split=2,\n           min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,\n           oob_score=False, random_state=None, verbose=0, warm_start=False))\n\npipe_llf_arousal = make_pipeline(\nRandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,\n           max_features=9, max_leaf_nodes=None, min_impurity_split=1e-07,\n           min_samples_leaf=1, min_samples_split=2,\n           min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,\n           oob_score=False, random_state=None, verbose=0, warm_start=False))")


# In[132]:

get_ipython().run_cell_magic(u'time', u'', u'import os\nidev_set = {}\nsmoothing = True\n\nallfold_metric=[]\nfolddf_dict={}\nmean_metric = []\n\nfor foldi in [1,2,3,4,5]:\n    trainlist, testlist=gettraintest2movielist(foldi,mov2groups)\n    os.system("mkdir NfoldCV/fold"+str(foldi))\n    testfolder="NfoldCV/fold"+str(foldi)+"/test/"\n    trainfolder="NfoldCV/fold"+str(foldi)+"/train/"\n    os.system("mkdir "+testfolder)\n    os.system("mkdir "+trainfolder)\n    \n    ii1,mse1,prs1,pvvis = trainPipe(foldi,pipe_visual_valence,\'MeanValence\',\'visual\')\n    ii2,mse2,prs2,pvaud = trainPipe(foldi,pipe_audio_valence,\'MeanValence\',\'audio\')\n    \n    ii3,mse3,prs3,pavis = trainPipe(foldi,pipe_visual_arousal,\'MeanArousal\',\'visual\')\n    ii4,mse4,prs4,paaud = trainPipe(foldi,pipe_audio_arousal,\'MeanArousal\',\'audio\')\n\n    ii5,mse5,prs5,pvdeep = trainPipe(foldi,pipe_deep_valence,\'MeanValence\',\'deep\')\n    ii6,mse6,prs6,padeep = trainPipe(foldi,pipe_deep_arousal,\'MeanArousal\',\'deep\')\n\n    ii7,mse7,prs7,pvlow = trainPipe(foldi,pipe_llf_valence,\'MeanValence\',\'llf\')\n    ii8,mse8,prs8,palow = trainPipe(foldi,pipe_llf_arousal,\'MeanArousal\',\'llf\')\n\n\n    fold_metric=[]\n    for f in testlist:\n        audiodf = getAudioDf(f)\n        visualdf = getAvgVisFeatListDf(f,visual_feat_list)\n        deepdf = getAvgVisFeatListDf(f,[\'fc6\'])\n        lowdf = getLowFeature10SecDf(f)        \n        \n        annotdf = getAnnotationDf(f)\n        ya = df2mat(annotdf[[\'MeanArousal\']])\n        yv = df2mat(annotdf[[\'MeanValence\']])\n        \n        print(audiodf.shape,visualdf.shape,deepdf.shape,lowdf.shape)\n\n        mlen = min([len(audiodf),len(visualdf),len(deepdf),len(lowdf)])\n\n        audiodf = audiodf[:mlen]\n        visualdf = visualdf[:mlen]\n        deepdf = deepdf[:mlen]\n        lowdf = lowdf[:mlen]\n        \n        ya = ya[:mlen]\n        yv = yv[:mlen]\n\n        aa = paaud.predict(audiodf)     \n        va = pvaud.predict(audiodf)\n        \n        av = pavis.predict(visualdf)\n        vv = pvvis.predict(visualdf)\n        \n        ad = padeep.predict(deepdf)\n        vd = pvdeep.predict(deepdf)\n        \n        al = palow.predict(lowdf)\n        vl = pvlow.predict(lowdf)\n        \n\n        if smoothing:\n            aa = holt_winters_second_order_ewma( aa, 10, 0.3 )\n            av = holt_winters_second_order_ewma( av, 10, 0.3 )\n            va = holt_winters_second_order_ewma( va, 10, 0.3 )\n            vv = holt_winters_second_order_ewma( vv, 10, 0.3 )\n            ad = holt_winters_second_order_ewma( ad, 10, 0.3 )\n            al = holt_winters_second_order_ewma( al, 10, 0.3 )\n            vd = holt_winters_second_order_ewma( vd, 10, 0.3 )\n            vl = holt_winters_second_order_ewma( vl, 10, 0.3 )\n        \n        mseaa, prsaa = getMetrics(ya,aa)\n        mseav, prsav = getMetrics(ya,av)\n        mseva, prsva = getMetrics(yv,va)\n        msevv, prsvv = getMetrics(yv,vv)\n\n        msead, prsad = getMetrics(ya,ad)\n        mseal, prsal = getMetrics(ya,al)\n        msevd, prsvd = getMetrics(yv,vd)\n        msevl, prsvl = getMetrics(yv,vl)\n\n        t = [msevv, prsvv[0], mseva, prsva[0] , mseaa, prsaa[0], mseav, prsav[0],\n             msevl, prsvl[0], msevd, prsvd[0] , mseal, prsal[0], msead, prsad[0]]\n                \n        fold_metric.append(t)\n        allfold_metric.append(t)\n\n        arousal_scores = np.transpose([ aa,av,ad,al ])\n        arousal_scores = np.mean(arousal_scores,axis=1)\n        valence_scores = np.transpose([va,vv,vd,vl ])\n        valence_scores = np.mean(valence_scores,axis=1)\n        \n        meandf = pd.DataFrame(np.transpose([valence_scores, arousal_scores]), columns=[\'MeanValence\',\'MeanArousal\'])\n\n        #mseA, prsA= getMetrics(ya,arousal_scores)\n        #mseV, prsV = getMetrics(yv,valence_scores)\n        \n        #mean_metric.append([mseV, prsV, mseA, prsA])\n\n        \n        df =pd.DataFrame(np.transpose([va, vv, vd, vl, aa, av, ad, al ]), \n                         columns=[\'MeanValenceAudio\',\'MeanValenceVisual\',\n                                  \'MeanValenceDeep\',\'MeanValenceLow\',\n                                  \'MeanArousalAudio\',\'MeanArousalVisual\',\n                                  \'MeanArousalDeep\',\'MeanArousalLow\'])\n        idev_set[f] = df\n        filename=testfolder+str(foldi)+"_"+f+".csv"\n        df.to_csv(filename, index=False)\n    \n    \n    folddf = pd.DataFrame(fold_metric, columns=[\'MeanValenceVisualMSE\',\'MeanValenceVisualPCC\',\n                                                \'MeanValenceAudioMSE\',\'MeanValenceAudioPCC\',\n                                                \'MeanArousalAudioMSE\',\'MeanArousalAudioPCC\',\n                                                \'MeanArousalVisualMSE\',\'MeanArousalVisualPCC\',\n                                                \'MeanValenceLowLevelMSE\',\'MeanValenceLowLevelPCC\',\n                                                \'MeanValenceDeepMSE\',\'MeanValenceDeepPCC\',\n                                                \'MeanArousalLowLevelMSE\',\'MeanArousalLowLevelPCC\',\n                                                \'MeanArousalDeepMSE\',\'MeanArousalDeepPCC\'])\n    \n    folddf.to_csv(testfolder+str(foldi)+"_metrics.csv") \n    folddf.describe().to_csv(testfolder+str(foldi)+"_metrics_stats.csv") \n    folddf_dict[foldi] = folddf\n    \n    #########################################ATTENTION ###############################\n    \'\'\' \n    for f in trainlist:\n        audiodf = getAudioDf(f)\n        visualdf = getAvgVisFeatListDf(f,visual_feat_list)\n        #print(audiodf.shape,visualdf.shape)\n\n        mlen = min(len(audiodf),len(visualdf))\n\n        audiodf = audiodf[:mlen]\n        visualdf = visualdf[:mlen]\n\n        aa = paa.predict(audiodf)\n        av = pav.predict(visualdf)\n        \n        va = pva.predict(audiodf)\n        vv = pvv.predict(visualdf)\n\n        if smoothing:\n            aa = holt_winters_second_order_ewma( aa, 10, 0.3 )\n            av = holt_winters_second_order_ewma( av, 10, 0.3 )\n            va = holt_winters_second_order_ewma( va, 10, 0.3 )\n            vv = holt_winters_second_order_ewma( vv, 10, 0.3 )\n\n        df =pd.DataFrame(np.transpose([ va, vv ,aa,av ]), columns=[\'MeanValenceAudio\',\'MeanValenceVisual\',\'MeanArousalAudio\',\'MeanArousalVisual\'])\n        idev_set[f] = df\n        filename=trainfolder+str(foldi)+"_"+f+".csv"\n        df.to_csv(filename, index=False)\n        \n    \'\'\'\n    \nallfolddf = pd.DataFrame(allfold_metric, columns=[\'MeanValenceVisualMSE\',\'MeanValenceVisualPCC\',\n                                                \'MeanValenceAudioMSE\',\'MeanValenceAudioPCC\',\n                                                \'MeanArousalAudioMSE\',\'MeanArousalAudioPCC\',\n                                                \'MeanArousalVisualMSE\',\'MeanArousalVisualPCC\',\n                                                \'MeanValenceLowLevelMSE\',\'MeanValenceLowLevelPCC\',\n                                                \'MeanValenceDeepMSE\',\'MeanValenceDeepPCC\',\n                                                \'MeanArousalLowLevelMSE\',\'MeanArousalLowLevelPCC\',\n                                                \'MeanArousalDeepMSE\',\'MeanArousalDeepPCC\'])\n    \nallfolddf.to_csv("all_metrics.csv") \nallfolddf.describe().to_csv("all_metrics_stats.csv")    \n#mean_metricdf=pd.DataFrame(mean_metric,columns=[\'mseV, prsV, mseA, prsA\'])')


# ## Evaluation results

# In[80]:

evaldf = pd.read_csv('all_metrics_stats.csv')


# In[81]:

evaldf.columns = [c.replace('Mean','') for c in evaldf.columns ]


# In[82]:

evaldf.set_index('Unnamed: 0',inplace=True)
evaldf.index.name = None


# In[83]:

armse = [f for f in evaldf.columns if ('Arousal' in f)  and ('MSE' in f) ]
arpcc = [f for f in evaldf.columns if ('Arousal' in f)  and ('PCC' in f) ]


# In[84]:

vlmse = [f for f in evaldf.columns if 'Valence' in f   and ('MSE' in f) ]
vlpcc = [f for f in evaldf.columns if 'Valence' in f   and ('PCC' in f) ]


# In[85]:

vlmse,armse


# In[86]:

evaldf[vlmse].transpose()[['mean','std']]


# In[87]:

evaldf[vlpcc].transpose()[['mean','std']]


# In[88]:

evaldf[armse].transpose()[['mean','std']]


# In[89]:

evaldf[arpcc].transpose()[['mean','std']]


# In[ ]:




# In[ ]:




# In[66]:


#test
f = "Wanted"
audiodf = getAudioDf(f)
visualdf = getAvgVisFeatListDf(f,visual_feat_list)
print(audiodf.shape,visualdf.shape)

mlen = min(len(audiodf),len(visualdf))

audiodf = audiodf[:mlen]
visualdf = visualdf[:mlen]

aa = paa.predict(audiodf)
av = pav.predict(visualdf)
arousal_scores = np.transpose([ aa,av ])
        

va = pvv.predict(audiodf)  ## look up this is twisted
vv = pva.predict(visualdf)
valence_scores = np.transpose([va,vv ])
        
df =pd.DataFrame(np.transpose([ aa,av , va,vv ])) #, columns=['MeanValence','MeanArousal'])


# ## Visualization
# 
# It looks like the pipe are successfully predict the movie "Decay", since it was in all the traiing sets.
# however 
# 

# In[185]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    dfa = getAnnotationDf(mov)
    dfa[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)
    dev_set[mov][['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)



# In[74]:

fix, axes = plt.subplots(figsize=(30,20))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    dfa = getAnnotationDf(mov)
    dfa[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)
    idev_set[mov].plot(ax=plt.gca(),title=mov,style=['g*-','mo-','y^-','bx-'])

