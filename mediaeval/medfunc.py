import numpy as np
import pylab as pl
import re, fileinput
import os.path
import glob

import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib

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

med2017home = '/home/deepuser/yasemin/'

# Input data
movieNames = ['After_The_Rain','Attitude_Matters','Barely_legal_stories','Between_Viewings','Big_Buck_Bunny','Chatter','Cloudland','Damaged_Kung_Fu','Decay','Elephant_s_Dream','First_Bite','Full_Service','Islands','Lesson_Learned','Norm','Nuclear_Family','On_time','Origami','Parafundit','Payload','Riding_The_Rails','Sintel','Spaceman','Superhero','Tears_of_Steel','The_room_of_franz_kafka','The_secret_number','To_Claire_From_Sonny','Wanted','You_Again']

pathcontinuous = med2017home
continuousAnnotationsFolder = pathcontinuous +'continuous-annotations/'
devdatacontinous =  pathcontinuous + "continuous-movies/"
#pathcontfeatures = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/Continuous/features-out-1/"

med2017visualFeaturesfolder= med2017home + 'MEDIAEVAL17-DevSet-Visual_features/features/'
med2017audiofolder= med2017home + 'MEDIAEVAL17-DevSet-Audio_features/features/'
med2017annotationsFolder = med2017home + 'MEDIAEVAL17-DevSet-Valence_Arousal-annotations/annotations/'
med2017fearFolder = med2017home + 'MEDIAEVAL17-DevSet-Fear-annotations/annotations/'

med2017visualFeaturesfolderTest= med2017home + 'MEDIAEVAL17-TestSet-Visual_features/visual_features/'
med2017audiofolderTest = med2017home + 'MEDIAEVAL17-TestSet-Audio_features/audio_features/'
med2017datafolderTest = med2017home + 'MEDIAEVAL17-TestSet-Data/data/'

groups = {
    0:['You_Again','Damaged_Kung_Fu','The_secret_number','Spaceman'],
    1:['Cloudland','Origami','Riding_The_Rails','Tears_of_Steel','Sintel'],
    2:['On_time','Elephant_s_Dream','Norm','Big_Buck_Bunny','Chatter','Full_Service'],
    3:['Islands','To_Claire_From_Sonny','Nuclear_Family','After_The_Rain','Parafundit'],
    4:['Decay'],
    5:['The_room_of_franz_kafka','Attitude_Matters','Lesson_Learned','Superhero'],
    6:['First_Bite','Wanted','Between_Viewings','Barely_legal_stories','Payload']
}

def gettraintestmovielist(groupno):
    testlist = groups[groupno]
    trainlist =[]
    for idx, group in enumerate(groups):
        if idx != groupno:
            for g in groups[idx]:
                trainlist.append(g)
    return trainlist, testlist


#Fear Annotations
def getFearDf(movname):
    filename = os.path.join(med2017fearFolder, movname + '-MEDIAEVAL2017-fear.txt')
    annotation = np.genfromtxt(filename, names=True, delimiter='\t', dtype=None)
    df = pd.DataFrame(annotation)
    return df

#Audio Features
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

def getMovieListAudioFearDf(movieNames,folder=med2017audiofolder):
    Xa = getAudioDf(movieNames[0],folder)
    y = getFearDf(movieNames[0]).Fear

    mlen = min(len(Xa),len(y))
    Xa = Xa[:mlen]
    y = y[:mlen]

    for mov in movieNames[1:]:
        tXa = getAudioDf(mov)
        ty = getFearDf(mov).Fear

        mlen = min(len(tXa),len(ty))
        tXa = tXa[:mlen]
        ty = ty[:mlen]

        Xa  = Xa.append(tXa)
        y  = y.append(ty)

    return Xa,y

def getVisFeatureDf(moviename,typename):
    files = glob.glob(med2017visualFeaturesfolder+moviename+'/'+typename+'/*.txt')
    files = sorted(files)
    alist = []
    for fname in files:
        f=open(fname,'r')
        for l in f:
            alist.append(map(float,l.split(',')))
        f.close()
    return pd.DataFrame(alist)

def getAvgVisFeatureDf(moviename,typename):
    df = getVisFeatureDf(moviename,typename)
    dfwindow = df.rolling(10).mean()[10::5]
    dfwindow.reset_index(inplace=True)
    dfwindow.drop('index',axis=1,inplace=True)
    return dfwindow

def getAvgVisFeatListDf(moviename,featlist):
    df = getVisFeatureDf(moviename,featlist[0])
    for feat in featlist[1:]:
        tdf = getVisFeatureDf(moviename,feat)
        df = pd.concat([df,tdf],axis=1)

    dfwindow = df.rolling(10).mean()[10::5]
    dfwindow.reset_index(inplace=True)
    dfwindow.drop('index',axis=1,inplace=True)
    dfwindow.columns=list(range(len(dfwindow.columns)))
    return dfwindow

def getMovListVisFearDf(movieNames,featname):
    X = getAvgVisFeatureDf(movieNames[0],featname)
    y = getFearDf(movieNames[0]).Fear[:len(X)]

    for mov in movieNames[1:]:
        tX = getAvgVisFeatureDf(mov,featname)
        ty = getFearDf(mov).Fear[:len(tX)]
        X  = X.append(tX)
        y  = y.append(ty)

    return X,y

def getMovListVisFusionFearDf(movieNames,featlist):
    X,y = getMovListVisFearDf(movieNames,featlist[0])
    # y is the target value and it is equal for all feature type
    for feattype in featlist[1:] :
        tX, y = getMovListVisFearDf(movieNames,feattype)
        X = pd.concat( [X,tX], axis=1 )
    return X,y

def getMovListAudioVisFearFeatListDf(movieNames,featlist):
    Xv = getAvgVisFeatureDf(movieNames[0],featlist[0])
    Xa = getAudioDf(movieNames[0])
    y = getFearDf(movieNames[0]).Fear

    mlen = min(len(Xv),len(Xa),len(y))

    Xv = Xv[:mlen]
    Xa = Xa[:mlen]
    y = y[:mlen]

    for feattype in featlist[1:]:
        fXv = getAvgVisFeatureDf(movieNames[0],feattype)[:mlen]
        Xv = pd.concat( [Xv,fXv], axis=1 )

    for mov in movieNames[1:]:
        tXv = getAvgVisFeatureDf(mov,featlist[0])
        tXa = getAudioDf(mov)
        ty = getFearDf(mov).Fear

        mlen = min(len(tXv),len(tXa),len(ty))
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

