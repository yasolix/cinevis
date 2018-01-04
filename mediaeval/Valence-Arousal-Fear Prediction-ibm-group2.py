
# coding: utf-8

# ###[alt text](/home/yt/datascience/The-circumplex.ppm.png "circumplex")
# 
# ## Valence-Arousal Prediction Audio and Visual Features
# 
# The Mediaeval 2017 Emotional Impact of Movies Task includes the data in the emotional domain 
# (valence - arousal  regression) and  fear ( binary classification).
# We have displayed the valence and arousal of all the movies in the dataset.
# Also the time of the movie where fear is present is specified with the value of the second.
# According to the Russell's circumplex model we were expectinf the "FEAR" to be appeared in the negative vallence, positive arousal part of the circumflex.
# However in some movies, we can see that frightment exists in positive valence with negative arousal also.
# 

# In[1]:


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


# In[2]:

import numpy as np 
print(np.__version__) 
print(np.__path__)


# In[3]:

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


# In[4]:

med2017visualFeaturesfolder='/home/deepuser/yasemin/MEDIAEVAL17-DevSet-Visual_features/features/'
med2017audiofolder='/home/deepuser/yasemin/MEDIAEVAL17-DevSet-Audio_features/features/'
med2017annotationsFolder = '/home/deepuser/yasemin/MEDIAEVAL17-DevSet-Valence_Arousal-annotations/annotations/'
med2017fearFolder = '/home/deepuser/yasemin/MEDIAEVAL17-DevSet-Fear-annotations/annotations/'
med2017dataFolder='/home/deepuser/yasemin/continuous-movies/'


# In[5]:

med2017visualfolderTest='/home/deepuser/yasemin/MEDIAEVAL17-TestSet-Visual_features/visual_features/'
med2017audiofolderTest = '/home/deepuser/yasemin/MEDIAEVAL17-TestSet-Audio_features/audio_features/'
med2017datafolderTest = '/home/deepuser/yasemin/MEDIAEVAL17-TestSet-Data/data/'


# In[6]:

files = glob.glob(med2017datafolderTest+'*')
testmovieNames =[ f.split('/')[-1].replace('.mp4','') for f in sorted(files) ]


# In[7]:

files = glob.glob(med2017dataFolder+'*')
movieNames =[ f.split('/')[-1].replace('.mp4','') for f in sorted(files) ]


# In[8]:

movieNames,testmovieNames


# In[ ]:




# In[9]:

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


# In[10]:

gettraintest2movielist(4)


# ## Valence - Arosal Annotations
# Thank you for downloading LIRIS-ACCEDE dataset.
# This file contains valence/arousal annotations for the LIRIS-ACCEDE continuous part that is used for the first subtask of the MEDIAEVAL 2017 Emotional Impact of Movies task.
# For each of the 30 movies, consecutive ten seconds-segments sliding over the whole movie with a shift of 5 seconds are considered and provided with valence and arousal annotations.
# Each txt file contains 4 columns separated by tabulations. The first column is the segment id, starting from 0, the second column is the starting time of the segment in the movie and the third and fourth columns are respectively the valence and arousal values for this segment.

# In[11]:

def getAnnotationDf(movname,folder=med2017annotationsFolder):
    filename = os.path.join(folder, movname + '-MEDIAEVAL2017-valence_arousal.txt')
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    return df


# In[12]:

#df = getAnnotationDf(movieNames[0])
#df.hist(alpha=0.5,bins=50)


# In[13]:

#df.head()
#df.describe()


# ## Valence, Arousal histogram plots for Dev-Set

# In[13]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(movieNames):
    if (ii+1 > 30):
        plt.subplot(6,5,ii)
    else :
        plt.subplot(6,5,ii+1)
    df = getAnnotationDf(mov)
    df[['MeanValence','MeanArousal']].plot.hist(ax=plt.gca(),title=mov,alpha=0.5,bins=50)


# ## Valence , Arousal plots for Dev-Set

# In[28]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    df = getAnnotationDf(mov)
    df[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)
    #.hist(alpha=0.5,bins=50)


# ## Fear Annotations

# In[14]:

def getFearDf(movname):
    filename = os.path.join(med2017fearFolder, movname + '-MEDIAEVAL2017-fear.txt')
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    return df


# In[14]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    df = getFearDf(mov)
    df[['Fear']].plot(ax=plt.gca(),title=mov)


# ## Audio Features
# 

# In[15]:

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

# In[16]:

visual_feat = ['acc', 'cedd', 'cl', 'eh', 'fc6', 'fcth', 
               'gabor', 'jcd', 'lbp', 'sc', 'tamura'   ]
visual_feat_wofc16 = ['acc', 'cedd', 'cl', 'eh', 'fcth', 
               'gabor', 'jcd', 'lbp', 'sc', 'tamura'   ]


# In[17]:

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
    dfwindow = df.rolling(10).mean()[10::5]
    dfwindow.reset_index(inplace=True)
    dfwindow.drop('index',axis=1,inplace=True)
    return dfwindow

def getAvgVisFeatListDf(moviename,featlist,folder=med2017visualFeaturesfolder):
    df = getVisFeatureDf(moviename,featlist[0],folder)
    for feat in featlist[1:]:
        tdf = getVisFeatureDf(moviename,feat,folder)
        df = pd.concat([df,tdf],axis=1)
    
    dfwindow = df.rolling(10).mean()[10::5]
    dfwindow.reset_index(inplace=True)
    dfwindow.drop('index',axis=1,inplace=True)
    dfwindow.columns=list(range(len(dfwindow.columns)))
    return dfwindow


# In[20]:

sum([len(getAnnotationDf(m)) for m in movieNames ])


# In[21]:

sum([len(getAudioDf(m)) for m in movieNames ])


# In[19]:

sum([len(getVisFeatureDf(m,'cl')) for m in movieNames ])


# In[20]:

sum([len(getAvgVisFeatureDf(m,'cl')) for m in movieNames ])


# In[ ]:

#df = getVisFeatureDf(movieNames[0],'cl')
#df = getAvgVisFeatureDf(movieNames[0],'cl')
#df = getAvgVisFeatListDf(movieNames[0],['cl','eh'])
#df.hist()


# In[18]:

#df = getAvgVisFeatListDf(movieNames[0],['fc6'])
#df = getVisFeatureDf(movieNames[0],'fc6')
#df.describe()


# ## Train and Test set creation

# In[18]:

def getFeatureswFearDf(movieNames,featlist=visual_feat_wofc16):
    Xv = getAvgVisFeatListDf(movieNames[0],featlist)
    Xa = getAudioDf(movieNames[0])
    Xd = getAvgVisFeatListDf(movieNames[0],['fc6'])
    y = getFearDf(movieNames[0])[['Fear']]
    
    mlen = min(len(Xv),len(Xa), len(Xd),len(y))
    
    Xv = Xv[:mlen]
    Xa = Xa[:mlen]
    Xd = Xd[:mlen]
    y = y[:mlen]
    
    for mov in movieNames[1:]:
        tXv = getAvgVisFeatListDf(mov,featlist)
        tXa = getAudioDf(mov)
        tXd = getAvgVisFeatListDf(mov,['fc6'])
        ty = getFearDf(mov)[['Fear']]
        
        mlen = min(len(tXv),len(tXa),len(tXd),len(ty))
        tXv = tXv[:mlen]
        tXa = tXa[:mlen]
        tXd = tXd[:mlen]
        ty = ty[:mlen]
        
        Xv  = Xv.append(tXv)
        Xa  = Xa.append(tXa)
        Xd = Xd.append(tXd)
        y  = y.append(ty)
        
    return Xv,Xa,Xd,y


# In[19]:

def getFeatureswAnnotationsDf(movieNames,featlist=visual_feat_wofc16):
    Xv = getAvgVisFeatListDf(movieNames[0],featlist)
    Xa = getAudioDf(movieNames[0])
    Xd = getAvgVisFeatListDf(movieNames[0],['fc6'])
    y = getAnnotationDf(movieNames[0])[['MeanValence','MeanArousal']]
    
    mlen = min(len(Xv),len(Xa), len(Xd),len(y))
    
    Xv = Xv[:mlen]
    Xa = Xa[:mlen]
    Xd = Xd[:mlen]
    y = y[:mlen]
    
    for mov in movieNames[1:]:
        tXv = getAvgVisFeatListDf(mov,featlist)
        tXa = getAudioDf(mov)
        tXd = getAvgVisFeatListDf(mov,['fc6'])
        ty = getAnnotationDf(mov)[['MeanValence','MeanArousal']]
        
        mlen = min(len(tXv),len(tXa),len(tXd),len(ty))
        tXv = tXv[:mlen]
        tXa = tXa[:mlen]
        tXd = tXd[:mlen]
        ty = ty[:mlen]
        
        Xv  = Xv.append(tXv)
        Xa  = Xa.append(tXa)
        Xd = Xd.append(tXd)
        y  = y.append(ty)
        
    return Xv,Xa,Xd,y


# In[20]:

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


# In[21]:

def df2mat(df):
    return df.as_matrix().reshape((len(df),))


# # Classification work

# In[21]:

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

from sklearn import cross_validation
from sklearn import metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams

def getGridCV(pipe,paramgirid,Xtrain,ytrain): # scoring ?
    grid = GridSearchCV(pipe, param_grid, cv=5,n_jobs=125)
    grid.fit(Xtrain,ytrain)
    
    return grid



# In[22]:

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


# In[23]:

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




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Regression work

# In[ ]:

#trainlist, testlist=gettraintestmovielist(2,movgroups_wodecay)  # index 1 olanları test , diğerlerini train yapan fonksiyon
#trainlist, testlist
#for ii in range(len(movgroups)):
#    trnlist, tstlist=gettraintestmovielist(ii)


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'allXv,allXa,allXd,ally = getFeatureswAnnotationsDf(movieNames)\nprint(allXv.shape,allXa.shape,allXd.shape,ally.shape)')


# In[25]:

trainlist, testlist=gettraintest2movielist(2)  # index 1 olanları test , diğerlerini train yapan fonksiyon

tXv,tXa,tXd,ty = getFeatureswAnnotationsDf(trainlist)
print(tXv.shape,tXa.shape,tXd.shape,ty.shape)
testXv, testXa, testXd, testy = getFeatureswAnnotationsDf(testlist)
print(testXv.shape, testXa.shape,testXd.shape, testy.shape)

X_train, X_test, y_train, y_test = train_test_split(tXv, ty,test_size=0.2, random_state=0)
Xa_train, Xa_test, ya_train, ya_test = train_test_split(tXa, ty,test_size=0.2, random_state=0)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(tXd, ty,test_size=0.2, random_state=0)


# In[26]:




# In[27]:

#ytrainarrayVal = ty[['MeanValence']].as_matrix().reshape((len(ty),))
#ytrainarrayAr = ty[['MeanArousal']].as_matrix().reshape((len(ty),))

#ytestarrayVal = testy[['MeanValence']].as_matrix().reshape((len(testy),))
#ytestarrayAr = testy[['MeanArousal']].as_matrix().reshape((len(testy),))


# In[ ]:




# In[114]:

#tXv,tXa,ty = getMovListAudioVisFeatListwAnnotationsDf(movieNames,['cl','gabor'])
#tXv.shape,tXa.shape,ty.shape


# In[28]:

#X_train, X_test, y_train, y_test = train_test_split(tXv, ty,test_size=0.32, random_state=0)

#Xa_train, Xa_test, ya_train, ya_test = train_test_split(tXa, ty,test_size=0.2, random_state=0)

#Xd_train, Xd_test, yd_train, yd_test = train_test_split(tXd, ty,test_size=0.2, random_state=0)


# In[29]:

#y_train.head()


# ## Linear Regression - Valence

# In[26]:

from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Create linear regression object
visual_regr = linear_model.LinearRegression()
audio_regr = linear_model.LinearRegression()
nn_regr = linear_model.LinearRegression()

# Train the model using the training sets
visual_regr.fit(X_train, y_train[['MeanValence']].as_matrix().reshape((len(y_train))))
audio_regr.fit(Xa_train, ya_train[['MeanValence']].as_matrix().reshape((len(ya_train))))
nn_regr.fit(Xd_train, yd_train[['MeanValence']].as_matrix().reshape((len(yd_train))))

# Make predictions using the testing set
visual_y_pred = visual_regr.predict(X_test)
audio_y_pred = audio_regr.predict(Xa_test)
nn_y_pred = nn_regr.predict(Xd_test)



# In[27]:

# The coefficients
print('Visual Coefficients: \n', visual_regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df2mat(y_test[['MeanValence']]), visual_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df2mat(y_test[['MeanValence']]),visual_y_pred))

print('pearson score  ',pearsonr(df2mat(y_test[['MeanValence']]),visual_y_pred))

print('Audio Coefficients: \n', audio_regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df2mat(ya_test[['MeanValence']]), audio_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df2mat(ya_test[['MeanValence']]),audio_y_pred))

print('pearson score  ',pearsonr(df2mat(ya_test[['MeanValence']]),audio_y_pred))

print('FC16 Coefficients: \n', nn_regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df2mat(yd_test[['MeanValence']]), nn_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df2mat(yd_test[['MeanValence']]),nn_y_pred))

print('pearson score  ',pearsonr(df2mat(yd_test[['MeanValence']]),nn_y_pred))


# ## Grid Search on Visual Features- Valence

# In[26]:

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


def getGridCV(pipe,paramgirid,Xtrain,ytrain): # scoring ?
    grid = GridSearchCV(pipe, param_grid, cv=5,n_jobs=125)
    grid.fit(Xtrain,ytrain)
    
    return grid

    


# In[33]:

get_ipython().run_cell_magic(u'time', u'', u'\n#X_train, X_test, y_train, y_test \n#pipe = Pipeline([(\'preprocessing\', StandardScaler()),(\'reduce_dim\', PCA()) ,(\'classifier\', SVR())])\npipe = Pipeline([(\'preprocessing\', StandardScaler()), (\'classifier\', SVR())])\n\nparam_grid = [\n    {\'classifier\': [SVR()], \n     \'preprocessing\': [StandardScaler()],\n#     \'reduce_dim\': [PCA()],\n#     \'reduce_dim__n_components\' : [ 800],\n     \'classifier__gamma\': [0.0001, 0.001,0.01, 0.1, 1, 10, 100],\n     \'classifier__C\': [0.001, 0.01, 0.1, 1, 10, 100,200]},\n    {\'classifier\': [RandomForestRegressor(n_estimators=100)],\n     \'preprocessing\': [None], \'classifier__max_features\': [3,5,10]}]\n\ngrid_vis_valence = getGridCV(pipe,param_grid,X_train,df2mat(y_train[[\'MeanValence\']]))\n\nprint("Best params:\\n{}\\n".format(grid_vis_valence.best_params_))\nprint("Best cross-validation score: {:.2f}".format(grid_vis_valence.best_score_))\nprint("All grid scores")\n\ngrid_vis_valence.grid_scores_, grid_vis_valence.best_params_, grid_vis_valence.best_score_')


# In[35]:

tXv.shape,tXa.shape,ty.shape,
testXv.shape, testXa.shape, testy.shape


# In[34]:

#scores[0]
#scores.to_csv('grid_vis_valence.txt')


# In[25]:

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


# In[38]:

gridscores(grid_vis_valence).head()


# ## Metrics and Paralell crossvalidation

# In[22]:

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


# In[23]:

def evaluate_pipe(pipe,trainX,trainy,testX,testy):
    
    ytrainarray = trainy.as_matrix().reshape((len(trainy),))
    ytestarray = testy.as_matrix().reshape((len(testy),))

    pipe.fit(trainX, ytrainarray)
    
    print("Train score: {:.2f}".format(pipe.score(trainX, ytrainarray)))
    print("Test score: {:.2f}".format(pipe.score(testX, ytestarray)))

    y_pred = pipe.predict(testX)

    mse, prs = getMetrics(ytestarray,y_pred)
    
    return y_pred,mse,prs[0],pipe


# In[24]:

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




# In[29]:

#trainlist, testlist=gettraintest2movielist(2,mov2groups)  # index 1 olanları test , diğerlerini train yapan fonksiyon
#trainlist


# In[42]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_visual_valence = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\npvv_scores = crossgroups(pipe_visual_valence,'MeanValence','visual')")


# In[43]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_visual_arousal = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\npav_scores = crossgroups(pipe_visual_arousal,'MeanArousal','visual')")


# In[36]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_audio_valence = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n)\npva_scores = crossgroups(pipe_audio_valence,'MeanValence','audio')")


# In[37]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_audio_arousal = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    #PCA(n_components=800),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n)\npaa_scores = crossgroups(pipe_audio_arousal,'MeanArousal','audio')")


# In[55]:

paa_scores.sort_values('PCC', ascending=False)


# In[56]:

pav_scores.sort_values('PCC', ascending=False)


# In[57]:

pva_scores.sort_values('PCC', ascending=False)


# In[58]:

pvv_scores.sort_values('PCC', ascending=False)


# ## Pipeline train

# In[27]:

get_ipython().run_cell_magic(u'time', u'', u"pipe_visual_valence = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\npipe_visual_arousal = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\npipe_audio_valence = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))\n\npipe_audio_arousal = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    #PCA(n_components=800),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))")


# In[28]:

ii1,mse1,prs1,pva = trainPipe(4,pipe_visual_valence,'MeanValence','visual')
ii2,mse2,prs2,pvv = trainPipe(3,pipe_audio_valence,'MeanValence','audio')



# In[29]:

ii3,mse3,prs3,pav = trainPipe(1,pipe_visual_arousal,'MeanArousal','visual')
ii4,mse4,prs4,paa = trainPipe(1,pipe_audio_arousal,'MeanArousal','audio')


# ## Smoothing

# In[31]:

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


# In[35]:

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
 


# In[72]:

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


# In[46]:

smooth1.shape,aa.shape


# ## Generating N-fold csv

# In[30]:

visual_feat_list= ['acc', 'cedd', 'cl', 'eh', 'fcth', 
               'gabor', 'jcd', 'lbp', 'sc', 'tamura'   ]


# In[ ]:

pipe_visual_valence = make_pipeline(
    StandardScaler(), 
    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, 
        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, 
        tol=0.001, verbose=False))

pipe_visual_arousal = make_pipeline(
    StandardScaler(), 
    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, 
        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, 
        tol=0.001, verbose=False))

pipe_audio_valence = make_pipeline(
    StandardScaler(copy=True, with_mean=True, with_std=True),
    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))

pipe_audio_arousal = make_pipeline(
    StandardScaler(copy=True, with_mean=True, with_std=True),
    #PCA(n_components=800),
    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))


# In[78]:

get_ipython().run_cell_magic(u'time', u'', u'import os\nidev_set = {}\nsmoothing = True\n\nallfold_metric=[]\nfolddf_dict={}\nmean_metric = []\n\nfor foldi in [1,2,3,4,5]:\n    trainlist, testlist=gettraintest2movielist(foldi,mov2groups)\n    os.system("mkdir NfoldCV/fold"+str(foldi))\n    testfolder="NfoldCV/fold"+str(foldi)+"/test/"\n    trainfolder="NfoldCV/fold"+str(foldi)+"/train/"\n    os.system("mkdir "+testfolder)\n    os.system("mkdir "+trainfolder)\n    \n    ii1,mse1,prs1,pvv = trainPipe(foldi,pipe_visual_valence,\'MeanValence\',\'visual\')\n    ii2,mse2,prs2,pva = trainPipe(foldi,pipe_audio_valence,\'MeanValence\',\'audio\')\n    \n    ii3,mse3,prs3,pav = trainPipe(foldi,pipe_visual_arousal,\'MeanArousal\',\'visual\')\n    ii4,mse4,prs4,paa = trainPipe(foldi,pipe_audio_arousal,\'MeanArousal\',\'audio\')\n\n\n    fold_metric=[]\n    for f in testlist:\n        audiodf = getAudioDf(f)\n        visualdf = getAvgVisFeatListDf(f,visual_feat_list)\n        annotdf = getAnnotationDf(f)\n        ya = df2mat(annotdf[[\'MeanArousal\']])\n        yv = df2mat(annotdf[[\'MeanValence\']])\n        \n        print(audiodf.shape,visualdf.shape)\n\n        mlen = min(len(audiodf),len(visualdf))\n\n        audiodf = audiodf[:mlen]\n        visualdf = visualdf[:mlen]\n        ya = ya[:mlen]\n        yv = yv[:mlen]\n\n        aa = paa.predict(audiodf)\n        av = pav.predict(visualdf)\n        \n        va = pva.predict(audiodf)\n        vv = pvv.predict(visualdf)\n        \n        if smoothing:\n            aa = holt_winters_second_order_ewma( aa, 10, 0.3 )\n            av = holt_winters_second_order_ewma( av, 10, 0.3 )\n            va = holt_winters_second_order_ewma( va, 10, 0.3 )\n            vv = holt_winters_second_order_ewma( vv, 10, 0.3 )\n        \n        mseaa, prsaa = getMetrics(ya,aa)\n        mseav, prsav = getMetrics(ya,av)\n        mseva, prsva = getMetrics(yv,va)\n        msevv, prsvv = getMetrics(yv,vv)\n        \n        fold_metric.append([msevv, prsvv[0], mseva, prsva[0] , mseaa, prsaa[0], mseav, prsav[0] ])\n        allfold_metric.append([msevv, prsvv[0], mseva, prsva[0] , mseaa, prsaa[0], mseav, prsav[0] ])\n\n        arousal_scores = np.transpose([ aa,av ])\n        arousal_scores = np.mean(arousal_scores,axis=1)\n        valence_scores = np.transpose([va,vv ])\n        valence_scores = np.mean(valence_scores,axis=1)\n        \n        meandf = pd.DataFrame(np.transpose([valence_scores, arousal_scores]), columns=[\'MeanValence\',\'MeanArousal\'])\n\n        mseA, prsA= getMetrics(ya,arousal_scores)\n        mseV, prsV = getMetrics(yv,valence_scores)\n        \n        mean_metric.append([mseV, prsV, mseA, prsA])\n\n        \n        df =pd.DataFrame(np.transpose([va, vv , aa, av ]), columns=[\'MeanValenceAudio\',\'MeanValenceVisual\',\'MeanArousalAudio\',\'MeanArousalVisual\'])\n        idev_set[f] = df\n        filename=testfolder+str(foldi)+"_"+f+".csv"\n        df.to_csv(filename, index=False)\n    \n    \n    folddf = pd.DataFrame(fold_metric, columns=[\'MeanValenceAudioMSE\',\'MeanValenceAudioPCC\',\n                                       \'MeanValenceVisualMSE\',\'MeanValenceVisualPCC\',\n                                       \'MeanArousalAudioMSE\',\'MeanArousalAudioPCC\',\n                                       \'MeanArousalVisualMSE\',\'MeanArousalVisualPCC\'])\n    \n    folddf.to_csv(testfolder+str(foldi)+"_metrics.csv") \n    folddf.describe().to_csv(testfolder+str(foldi)+"_metrics_stats.csv") \n    folddf_dict[foldi] = folddf\n    \n    #########################################ATTENTION ###############################\n    for f in trainlist:\n        audiodf = getAudioDf(f)\n        visualdf = getAvgVisFeatListDf(f,visual_feat_list)\n        #print(audiodf.shape,visualdf.shape)\n\n        mlen = min(len(audiodf),len(visualdf))\n\n        audiodf = audiodf[:mlen]\n        visualdf = visualdf[:mlen]\n\n        aa = paa.predict(audiodf)\n        av = pav.predict(visualdf)\n        \n        va = pva.predict(audiodf)\n        vv = pvv.predict(visualdf)\n\n        if smoothing:\n            aa = holt_winters_second_order_ewma( aa, 10, 0.3 )\n            av = holt_winters_second_order_ewma( av, 10, 0.3 )\n            va = holt_winters_second_order_ewma( va, 10, 0.3 )\n            vv = holt_winters_second_order_ewma( vv, 10, 0.3 )\n\n        df =pd.DataFrame(np.transpose([ va, vv ,aa,av ]), columns=[\'MeanValenceAudio\',\'MeanValenceVisual\',\'MeanArousalAudio\',\'MeanArousalVisual\'])\n        idev_set[f] = df\n        filename=trainfolder+str(foldi)+"_"+f+".csv"\n        df.to_csv(filename, index=False)\n        \n    \nallfolddf = pd.DataFrame(allfold_metric, columns=[\'MeanValenceAudioMSE\',\'MeanValenceAudioPCC\',\n                                       \'MeanValenceVisualMSE\',\'MeanValenceVisualPCC\',\n                                       \'MeanArousalAudioMSE\',\'MeanArousalAudioPCC\',\n                                       \'MeanArousalVisualMSE\',\'MeanArousalVisualPCC\'])\n    \nallfolddf.to_csv("all_metrics.csv") \nallfolddf.describe().to_csv("all_metrics_stats.csv")    \nmean_metricdf=pd.DataFrame(mean_metric,columns=[\'mseV, prsV, mseA, prsA\'])')


# ## Test-set RUN

# In[80]:

get_ipython().run_cell_magic(u'time', u'', u'\ndef getAVmeanscore(aa,av,va,vv):\n    arousal_scores = np.transpose([ aa,av ])\n    arousal_scores = np.mean(arousal_scores,axis=1)\n    valence_scores = np.transpose([va,vv ])\n    valence_scores = np.mean(valence_scores,axis=1)\n    \n    df = pd.DataFrame(np.transpose([valence_scores, arousal_scores]), columns=[\'MeanValence\',\'MeanArousal\'])\n    \n    return df\n\n\ndef getAVTestprediction(f):\n    \n    audiodf = getAudioDf(f,folder=med2017audiofolderTest)\n    visualdf = getAvgVisFeatListDf(f,visual_feat_list,\n                                   folder=med2017visualfolderTest)\n    print(audiodf.shape,visualdf.shape)\n    \n    mlen = min(len(audiodf),len(visualdf))\n    \n    audiodf = audiodf[:mlen]\n    visualdf = visualdf[:mlen]\n    \n    aa = paa.predict(audiodf)\n    av = pav.predict(visualdf)\n        \n    va = pvv.predict(audiodf)\n    vv = pva.predict(visualdf)\n\n    df = pd.DataFrame(np.transpose([vv, va , aa, av ]), columns=[\'MeanValenceAudio\',\'MeanValenceVisual\',\'MeanArousalAudio\',\'MeanArousalVisual\'])\n     \n    meandf = getAVmeanscore(aa,av,va,vv)\n    \n    return df\n\nos.system("mkdir ./Test-Prediction")\nitest_set = {}\n\nii1,mse1,prs1,pvv = trainPipe(4,pipe_visual_valence,\'MeanValence\',\'visual\')\nii2,mse2,prs2,pva = trainPipe(3,pipe_audio_valence,\'MeanValence\',\'audio\')\n   \nii3,mse3,prs3,pav = trainPipe(1,pipe_visual_arousal,\'MeanArousal\',\'visual\')\nii4,mse4,prs4,paa = trainPipe(1,pipe_audio_arousal,\'MeanArousal\',\'audio\')\n\nfname = \'me17ei_BOUNNKU_valence_arousal_smooth_2.txt\'\nfd = open(fname, \'w\')\nfd.close()\n\nfor f in testmovieNames:\n    print(f)\n    \n    fd = open(fname, \'a\')\n    fd.write(f+\'\\n\')  # python will convert \\n to os.linesep\n    fd.close()\n\n    audiodf = getAudioDf(f,folder=med2017audiofolderTest)\n    visualdf = getAvgVisFeatListDf(f,visual_feat_list,\n                                   folder=med2017visualfolderTest)\n    print(audiodf.shape,visualdf.shape)\n    \n    mlen = min(len(audiodf),len(visualdf))\n    \n    audiodf = audiodf[:mlen]\n    visualdf = visualdf[:mlen]\n    \n    aa = paa.predict(audiodf)\n    av = pav.predict(visualdf)\n        \n    va = pva.predict(audiodf)\n    vv = pvv.predict(visualdf)\n\n    if smoothing:\n        aa = holt_winters_second_order_ewma( aa, 10, 0.3 )\n        av = holt_winters_second_order_ewma( av, 10, 0.3 )\n        va = holt_winters_second_order_ewma( va, 10, 0.3 )\n        vv = holt_winters_second_order_ewma( vv, 10, 0.3 )\n\n    df = pd.DataFrame(np.transpose([vv, va , aa, av ]), columns=[\'MeanValenceAudio\',\'MeanValenceVisual\',\'MeanArousalAudio\',\'MeanArousalVisual\'])\n     \n    meandf = getAVmeanscore(aa,av,va,vv)\n    itest_set[f]=meandf  \n    #scoresdf , meanddf = getAVTestprediction(f) \n    sfilename="./Test-Prediction/"+f+".csv"\n    df.to_csv(sfilename, index=False)\n    \n    meandf.to_csv(fname, sep=\' \', index=False, header=False, mode=\'a\')\n    ')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




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


# In[166]:

#pd.DataFrame(results,columns=['test-group','MSE','PCC'])
pipe1scores = crossgroups(pipe_visual_valence,'MeanValence','visual')
pipe1scores


# In[42]:

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(Xa_train.shape, Xa_test.shape, ya_train.shape, ya_test.shape)
print(tXv.shape,tXa.shape,ty.shape)
print(testXv.shape, testXa.shape, testy.shape)


# In[43]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\n#X_train.shape, X_test.shape, y_train.shape, y_test.shape \n#tXv.shape,tXa.shape,ty.shape\n#testXv.shape, testXa.shape, testy.shape\n\npipe_visual_valence = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\n#SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n#  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n# SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n#  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n\n#X_train, X_test, y_train, y_test \ny_pred_visV_train,mse,prs,pipe_vv = evaluate_pipe(pipe_visual_valence, \n                                   X_train,y_train[['MeanValence']], \n                                   X_test,  y_test[['MeanValence']])\n#y_pred_vis,mse,prs = evaluate_pipe(pipe_visual_valence,tXv,ytrain,testXv,ytest)\n\ny_pred_visV_test,mse,prs,pipe_vv = evaluate_pipe(pipe_visual_valence, \n                                   tXv,ty[['MeanValence']], \n                                   testXv, testy[['MeanValence']])\n")


# In[ ]:




# In[55]:




# In[ ]:

#pipe_visual_valence_scores


# In[ ]:

#sorted(pipe_visual_valence_scores.MSE),sorted(pipe_visual_valence_scores.PCC)


# ## Grid-Search on Audio features - Valence

# In[48]:

get_ipython().run_cell_magic(u'time', u'', u'from sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nfrom sklearn.svm import SVR\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.grid_search import GridSearchCV \n\n#pipe = Pipeline([(\'preprocessing\', StandardScaler()),(\'reduce_dim\', PCA(n_components=600)) ,(\'classifier\', SVR())])\npipe = Pipeline([(\'preprocessing\', StandardScaler()),(\'classifier\', SVR())])\n\nparam_grid = [\n    {\'classifier\': [SVR()], \n     \'preprocessing\': [StandardScaler()],\n     #\'reduce_dim__n_components\' : [ 50, 100, 500 ,750],\n     \'classifier__gamma\': [0.001,0.01, 0.1, 1, 10, 100],\n     \'classifier__C\': [0.001, 0.01, 0.1, 1, 10, 100]},\n    {\'classifier\': [RandomForestRegressor()],\n     \'preprocessing\': [None],\n     \'classifier__n_estimators\': [100,400],\n     \'classifier__max_features\': [3,5,10]}\n]\n\ngrid_audio_valence = GridSearchCV(pipe, param_grid, cv=5,n_jobs=-1)\nytrainarray = ya_train[[\'MeanValence\']].as_matrix()\nytrainarray = ytrainarray.reshape((len(ya_train),))\ngrid_audio_valence.fit(Xa_train,ytrainarray)\n\nprint("Best params:\\n{}\\n".format(grid_audio_valence.best_params_))\nprint("Best cross-validation score: {:.2f}".format(grid_audio_valence.best_score_))\nprint("All grid scores")\n\ngrid_audio_valence.grid_scores_, grid_audio_valence.best_params_, grid_audio_valence.best_score_')


# In[63]:

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(Xa_train.shape, Xa_test.shape, ya_train.shape, ya_test.shape)
print(tXv.shape,tXa.shape,ty.shape)
print(testXv.shape, testXa.shape, testy.shape)


# In[49]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\npipe_audio_valence = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n)\n\n#    SVR(C=10, cache_size=200, coef0=0.0, degree=3, \n#       epsilon=0.1, gamma=0.001,\n#      kernel='rbf', max_iter=-1, \n#     shrinking=True, tol=0.001, verbose=False)\n\n\n#Xa_train, Xa_test, ya_train, ya_test \ny_pred_audV_train,mse,prs,pipe_va = evaluate_pipe(pipe_audio_valence, \n                                   Xa_train,ya_train[['MeanValence']], \n                                   Xa_test,ya_test[['MeanValence']])\n\ny_pred_audV_test,mse,prs,pipe_va = evaluate_pipe(pipe_audio_valence, \n                                   tXa,ty[['MeanValence']], \n                                   testXa,testy[['MeanValence']])\n")


# In[68]:

pipe2scores = crossgroups(pipe_audio_valence,'MeanValence','audio')


# In[69]:

pipe2scores


# In[70]:

pipe1scores


# In[149]:

y_pred_visV_train.shape,y_test.shape, y_pred_visV_test.shape,testy.shape


# In[151]:

y_pred_audV_train.shape,ya_test.shape, y_pred_audV_test.shape,testy.shape


# In[ ]:

y_val_vitrain = pipe_visual_valence.fit(tXv,df2mat(ty[['MeanValence']])).predict(tXv)
y_val_autrain = pipe_audio_valence.fit(tXa,df2mat(ty[['MeanValence']])).predict(tXa)

train_val = np.transpose([y_val_vitrain,y_val_autrain])
train_val_y = df2mat(ty[['MeanValence']])

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVR())])

param_grid = [
    {'classifier': [SVR()],
     'classifier__kernel':['linear','rbf'],
     'preprocessing': [StandardScaler()],
     'classifier__gamma': [0.001,0.01, 0.1, 1, 10, 100],
     'classifier__C': [ 1, 10, 100]}
]

grid_valence_fuse = GridSearchCV(pipe, param_grid, cv=5,n_jobs=100)
grid_valence_fuse.fit(train_val,train_val_y)

print("Best params:\n{}\n".format(grid_valence_fuse.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_valence_fuse.best_score_))
print("All grid scores")


# In[142]:

plt.figure(figsize=(20,20))
plt.scatter(range(len(testy)),y_pred_vis,c='r')
plt.scatter(range(len(testy)),y_pred_aud,c='b')
plt.scatter(range(len(testy)),df2mat(testy[['MeanValence']]),c='g')


# ## Grid Search on Visual Features -Arousal

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u"pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVR())])\n\nparam_grid = [\n    {'classifier': [SVR()], 'preprocessing': [StandardScaler()],\n     'classifier__gamma': [0.001,0.01, 0.1, 1, 10, 100],\n     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},\n    {'classifier': [RandomForestRegressor()],\n     'preprocessing': [None],\n     'classifier__n_estimators': [100,400, 600, 1000],\n     'classifier__max_features': [3,5,10,15,20]}\n]\n\ngrid_vis_arousal = GridSearchCV(pipe, param_grid, cv=5,n_jobs=120)\nytrainarray = y_train[['MeanArousal']].as_matrix()\nytrainarray = ytrainarray.reshape((len(y_train),))\ngrid_vis_arousal.fit(X_train,ytrainarray)\n")


# In[72]:

print("Best params:\n{}\n".format(grid_vis_arousal.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_vis_arousal.best_score_))
grid_vis_arousal.grid_scores_, grid_vis_arousal.best_params_, grid_vis_arousal.best_score_


# In[73]:

grid_vis_arousal.best_params_


# In[74]:

grid_vis_arousal.best_score_


# In[75]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\n#X_train.shape, X_test.shape, y_train.shape, y_test.shape \n#tXv.shape,tXa.shape,ty.shape\n#testXv.shape, testXa.shape, testy.shape\n\npipe_visual_arousal = make_pipeline(\n    StandardScaler(), \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, \n        gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True, \n        tol=0.001, verbose=False))\n\n#X_train, X_test, y_train, y_test \ny_pred_visA_train,mse,prs,pipe_av = evaluate_pipe(pipe_visual_arousal, \n                                   X_train,y_train[['MeanArousal']], \n                                   X_test,  y_test[['MeanArousal']])\n\ny_pred_visA_test,mse,prs,pipe_av = evaluate_pipe(pipe_visual_arousal, \n                                   tXv,ty[['MeanArousal']], \n                                   testXv, testy[['MeanArousal']])")


# In[148]:

y_pred_visA_train.shape,y_test.shape, y_pred_visA_test.shape,testy.shape


# In[76]:

pipe3scores = crossgroups(pipe_visual_arousal,'MeanArousal','visual')


# In[77]:

pipe3scores


# ## Grid Search on Audio Features -Arousal

# In[78]:

get_ipython().run_cell_magic(u'time', u'', u"pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVR())])\n\nparam_grid = [\n    {'classifier': [SVR()], 'preprocessing': [StandardScaler()],\n     'classifier__gamma': [0.001,0.01, 0.1, 1, 10, 100],\n     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},\n    {'classifier': [RandomForestRegressor()],\n     'preprocessing': [None],\n     'classifier__n_estimators': [100,400, 600, 1000],\n     'classifier__max_features': [3,5,10,15,20]}\n]\n\ngrid_audio_arousal = GridSearchCV(pipe, param_grid, cv=5,n_jobs=220)\nytrainarray = ya_train[['MeanArousal']].as_matrix()\nytrainarray = ytrainarray.reshape((len(ya_train),))\ngrid_audio_arousal.fit(Xa_train,ytrainarray)\n")


# In[79]:

print("Best params:\n{}\n".format(grid_audio_arousal.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_audio_arousal.best_score_))
grid_audio_arousal.grid_scores_, grid_audio_arousal.best_params_, grid_audio_arousal.best_score_


# In[80]:

grid_audio_arousal.best_score_


# In[148]:

X_train.columns


# In[81]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\npipe_audio_arousal = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),  \n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n)\ny_pred_audA_train,mse,prs,pipe_aa = evaluate_pipe(pipe_audio_arousal, \n                                   Xa_train,ya_train[['MeanArousal']], \n                                   Xa_test,ya_test[['MeanArousal']])\n\ny_pred_audA_test,mse,prs,pipe_aa = evaluate_pipe(pipe_audio_arousal, \n                                   tXa,ty[['MeanArousal']], \n                                   testXa,testy[['MeanArousal']])\n")


# In[93]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA, NMF\nfrom sklearn.feature_selection import SelectKBest, chi2\n\npipe_audio_arousal = make_pipeline(\n    StandardScaler(copy=True, with_mean=True, with_std=True),\n    #PCA(n_components=800),\n    SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,\n        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n)\ny_pred_audA_train,mse,prs,pipe_aa2 = evaluate_pipe(pipe_audio_arousal, \n                                   Xa_train,ya_train[['MeanArousal']], \n                                   Xa_test,ya_test[['MeanArousal']])\n\ny_pred_audA_test,mse,prs,pipe_aa2 = evaluate_pipe(pipe_audio_arousal, \n                                   tXa,ty[['MeanArousal']], \n                                   testXa,testy[['MeanArousal']])\n")


# In[82]:

pipe4scores = crossgroups(pipe_audio_arousal,'MeanArousal','audio')


# In[141]:

pipe4scores


# In[83]:

y_pred_visA_train.shape,y_test.shape, y_pred_visA_test.shape,testy.shape


# In[84]:

y_pred_audA_train.shape,ya_test.shape, y_pred_audA_test.shape,testy.shape


# In[93]:

y_pred_meanV = np.mean(np.transpose([y_pred_visV_test,y_pred_audV_test]),axis=1)


# In[154]:

pipe11=pipe_visual_valence
pipe22=pipe_audio_valence
pipe33=pipe_visual_arousal
pipe44=pipe_audio_arousal


# In[ ]:

allXv,allXa,allXd,ally = getFeatureswAnnotationsDf(movieNames)



# In[155]:

print(allXv.shape,allXa.shape,allXd.shape,ally.shape)


# In[82]:

allXv_train, allXv_test, ally_train, ally_test = train_test_split(allXv, ally,test_size=0.2, random_state=0)
allXa_train, allXa_test, allya_train, allya_test = train_test_split(allXa, ally,test_size=0.2, random_state=0)
allXd_train, allXd_test, allyd_train, allyd_test = train_test_split(allXd, ally,test_size=0.2, random_state=0)


# In[167]:

get_ipython().run_cell_magic(u'time', u'', u"y_pred_aa_test,mse,prs,pipe4aa = evaluate_pipe(pipe_audio_arousal, \n                                   allXa_train,allya_train[['MeanArousal']], \n                                   testXa,testy[['MeanArousal']])\n")


# In[168]:

get_ipython().run_cell_magic(u'time', u'', u"y_pred_av_test,mse,prs,pipe3av = evaluate_pipe(pipe_visual_arousal, \n                                   allXv_train,ally_train[['MeanArousal']], \n                                   testXv,testy[['MeanArousal']])\n")


# In[169]:

y_pred_va_test,mse,prs,pipe2va = evaluate_pipe(pipe_audio_valence, 
                                   allXa_train,allya_train[['MeanValence']], 
                                   testXa,testy[['MeanValence']])
y_pred_vv_test,mse,prs,pipe1vv = evaluate_pipe(pipe_visual_valence, 
                                   allXv_train,ally_train[['MeanValence']], 
                                   testXv,testy[['MeanValence']])


# In[170]:

pipe4scores.sort_values('PCC')


# In[171]:

plt.figure(figsize=(20,20))
plt.scatter(range(len(testy)),y_pred_va_test,c='y')
plt.scatter(range(len(testy)),y_pred_vv_test,c='r')
plt.scatter(range(len(testy)),df2mat(testy[['MeanValence']]),c='g')


# In[172]:

plt.figure(figsize=(20,20))
plt.scatter(range(len(testy)),y_pred_aa_test,c='y')
plt.scatter(range(len(testy)),y_pred_av_test,c='r')
plt.scatter(range(len(testy)),df2mat(testy[['MeanArousal']]),c='g')


# In[94]:

plt.figure(figsize=(20,20))
#plt.scatter(range(len(testy)),y_pred_meanV,c='y')
#plt.scatter(range(len(testy)),y_pred_visV_test,c='r')
#plt.scatter(range(len(testy)),y_pred_audV_test,c='b')
plt.scatter(range(len(testy)),df2mat(testy[['MeanValence']]),c='g')


# In[95]:

y_pred_meanA = np.mean(np.transpose([y_pred_visA_test,y_pred_audA_test]),axis=1)


# In[96]:

plt.figure(figsize=(20,20))
plt.scatter(range(len(testy)),y_pred_meanA,c='y')
#plt.scatter(range(len(testy)),y_pred_visA_test,c='r')
#plt.scatter(range(len(testy)),y_pred_audA_test,c='b')
plt.scatter(range(len(testy)),df2mat(testy[['MeanArousal']]),c='g')


# # Test Run 1

# In[190]:

print(pipe_audio_arousal, 
pipe_visual_arousal, 
pipe_audio_valence, 
pipe_visual_valence)


# In[101]:

visual_feat_list= ['acc', 'cedd', 'cl', 'eh', 'fcth', 
               'gabor', 'jcd', 'lbp', 'sc', 'tamura'   ]


# In[187]:

get_ipython().run_cell_magic(u'time', u'', u"#pipe4aa\n#pipe3av\n#pipe2va\n#pipe1vv\nfname = 'me17ei_BOUNNKU_valence_arousal_1_run2.txt'\nfd = open(fname, 'w')\nfd.close()\n\ntest_set = {}\nfor f in testmovieNames:\n    print(f)\n    \n    fd = open(fname, 'a')\n    fd.write(f+'\\n')  # python will convert \\n to os.linesep\n    fd.close()\n    \n    audiodf = getAudioDf(f,folder=med2017audiofolderTest)\n    visualdf = getAvgVisFeatListDf(f,visual_feat_list,\n                                   folder=med2017visualfolderTest)\n    print(audiodf.shape,visualdf.shape)\n    \n    mlen = min(len(audiodf),len(visualdf))\n    \n    audiodf = audiodf[:mlen]\n    visualdf = visualdf[:mlen]\n    \n    aa = pipe4aa.predict(audiodf)\n    av = pipe3av.predict(visualdf)\n    arousal_scores = np.transpose([ aa,av ])\n    arousal_scores = np.mean(arousal_scores,axis=1)\n    \n    va = pipe2va.predict(audiodf)\n    vv = pipe1vv.predict(visualdf)\n    valence_scores = np.transpose([va,vv ])\n    valence_scores = np.mean(valence_scores,axis=1)\n    \n    df =pd.DataFrame(np.transpose([valence_scores, arousal_scores]), columns=['MeanValence','MeanArousal'])\n    df.to_csv(fname, sep=' ', index=False, header=False, mode='a')\n    test_set[f] = df\n    ")


# In[184]:

# ploting results on dev-
dev_set = {}
for f in movieNames:
    audiodf = getAudioDf(f)
    visualdf = getAvgVisFeatListDf(f,visual_feat_list)
    print(audiodf.shape,visualdf.shape)
    
    mlen = min(len(audiodf),len(visualdf))
    
    audiodf = audiodf[:mlen]
    visualdf = visualdf[:mlen]
    
    aa = pipe4aa.predict(audiodf)
    av = pipe3av.predict(visualdf)
    arousal_scores = np.transpose([ aa,av ])
    arousal_scores = np.mean(arousal_scores,axis=1)
    
    va = pipe2va.predict(audiodf)
    vv = pipe1vv.predict(visualdf)
    valence_scores = np.transpose([va,vv ])
    valence_scores = np.mean(valence_scores,axis=1)
    
    df =pd.DataFrame(np.transpose([valence_scores, arousal_scores]), columns=['MeanValence','MeanArousal'])
    dev_set[f] = df


# In[185]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    dfa = getAnnotationDf(mov)
    dfa[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)
    dev_set[mov][['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)



# In[81]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(testmovieNames):
    plt.subplot(5,3,ii+1)
    itest_set[mov][['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)


# In[188]:

fix, axes = plt.subplots(figsize=(20,16))
for ii, mov in enumerate(testmovieNames):
    plt.subplot(5,3,ii+1)
    test_set[mov][['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)



# ## Test 2- pipes with 2 groups
# 
# It looks like the pipe are successfully predict the movie "Decay", since it was in all the traiing sets.
# however 
# 

# In[74]:

fix, axes = plt.subplots(figsize=(30,20))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    dfa = getAnnotationDf(mov)
    dfa[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)
    idev_set[mov].plot(ax=plt.gca(),title=mov,style=['g*-','mo-','y^-','bx-'])


# In[56]:

fix, axes = plt.subplots(figsize=(30,20))
for ii, mov in enumerate(movieNames):
    plt.subplot(6,5,ii+1)
    dfa = getAnnotationDf(mov)
    dfa[['MeanValence','MeanArousal']].plot(ax=plt.gca(),title=mov)
    idev_set[mov].plot(ax=plt.gca(),title=mov,legend=None)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ## this run have some bug related to pipes 

# In[138]:

get_ipython().run_cell_magic(u'time', u'', u"fname = 'me17ei_BOUNNKU_valence_arousal_1.txt'\nfd = open(fname, 'w')\nfd.close()\n\nfor f in testmovieNames:\n    print(f)\n    \n    fd = open(fname, 'a')\n    fd.write(f+'\\n')  # python will convert \\n to os.linesep\n    fd.close()\n    \n    audiodf = getAudioDf(f,folder=med2017audiofolderTest)\n    visualdf = getAvgVisFeatListDf(f,visual_feat_list,\n                                   folder=med2017visualfolderTest)\n    print(audiodf.shape,visualdf.shape)\n    \n    mlen = min(len(audiodf),len(visualdf))\n    \n    audiodf = audiodf[:mlen]\n    visualdf = visualdf[:mlen]\n    \n    aa = pipe_audio_arousal.predict(audiodf)\n    av = pipe_visual_arousal.predict(visualdf)\n    arousal_scores = np.transpose([ aa,av ])\n    arousal_scores = np.mean(arousal_scores,axis=1)\n    \n    va = pipe_audio_valence.predict(audiodf)\n    vv = pipe_visual_valence.predict(visualdf)\n    valence_scores = np.transpose([va,vv ])\n    valence_scores = np.mean(valence_scores,axis=1)\n    \n    df =pd.DataFrame(np.transpose([valence_scores, arousal_scores]), columns=['MeanValence','MeanArousal'])\n    df.to_csv(fname, sep=' ', index=False, header=False, mode='a')\n    ")


# In[116]:

df =pd.DataFrame(np.transpose([valence_scores, arousal_scores]), columns=['MeanValence','MeanArousal'])


# In[125]:

df.to_csv('M00.txt', sep=' ', index=False, header=False, mode='a')


# 

# In[179]:




# In[142]:

get_ipython().system(u'ls m*')


# In[ ]:




# In[ ]:



