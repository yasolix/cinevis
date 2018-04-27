
# coding: utf-8

# In[1]:

import os
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import pearsonr

testset = ['MEDIAEVAL17_00','MEDIAEVAL17_01','MEDIAEVAL17_02','MEDIAEVAL17_03','MEDIAEVAL17_04','MEDIAEVAL17_05','MEDIAEVAL17_06','MEDIAEVAL17_07','MEDIAEVAL17_08','MEDIAEVAL17_09','MEDIAEVAL17_10','MEDIAEVAL17_11','MEDIAEVAL17_12','MEDIAEVAL17_13' ]

dir_input ='fusedir/'
dir_results = 'teams/'
dir_gt = "ground_truth/"


def read_groundtruth():
    names_gt = []
    valence_gt = []
    arousal_gt = []

    for test in testset:

        names_gt.append(test)

        valence = []
        arousal = []

        with open('ground_truth/' + test + '_Valence-Arousal.txt', 'r') as f:
            f.readline()  # Skip first line
            for line in f:
                lineParts = line.split()
                valence.append(float(lineParts[2]))
                arousal.append(float(lineParts[3]))

        valence_gt.append(valence)
        arousal_gt.append(arousal)

    return names_gt, valence_gt, arousal_gt


def read_valence_arousal(): 
    dict_val={}
    dict_ar={}
    files = os.listdir(dir_input)
    for fname in files:
        if fname.find('.txt') >= 0:
            print(fname)

            with open(dir_input+fname,'r') as f:
                lines = f.read().splitlines()
                #print(lines)

            dict_val[fname] ={}
            dict_ar[fname] ={}
            names=[]
            indexes = []
            for i in range(len(lines)):
                if lines[i].find("MEDIAEVAL17") >= 0:
                    indexes.append(i)
                    names.append(lines[i])

            for i in range(len(names)):
                ibeg = indexes[i]+1
                iend = indexes[i+1]-1 if i < len(names)-1 else len(lines)-1
                #print(ibeg,' ',iend)

                valence=[]
                arousal=[]
                for j in range(ibeg,iend+1):
                    items = lines[j].split()
                    valence.append(float(items[0]))
                    arousal.append(float(items[1]))

                dict_val[fname][names[i]] = valence
                dict_ar[fname][names[i]] = arousal
    
    return dict_val,dict_ar


def mean_fuse(dict_val,dict_ar,filename='fused.txt'):
    f = open(dir_results+filename,'w')
    for test in testset:
        #print(test)
        f.write(test+os.linesep)
        minv = min([ len(dict_val[i][test]) for i in dict_val])
        fusedval = np.mean([ dict_val[i][test][0:minv] for i in dict_val],axis=0)

        fusedar = np.mean([ dict_ar[i][test][0:minv] for i in dict_ar],axis=0)
        for i in range(len(fusedval)):
            f.write(str(fusedval[i])+' '+str(fusedar[i])+os.linesep)
    f.close()   

def mean_weighted_fuse(dict_val,dict_ar,weights=[1./4, 3./4],filename='fused_weighted.txt'):
    f = open(dir_results+filename,'w')
    fusedvallist=[]
    fusedarlist=[]
    for test in testset:
        #print(test)
        f.write(test+os.linesep)
        minv = min([ len(dict_val[i][test]) for i in dict_val])
        fusedval = np.average([ dict_val[i][test][0:minv] for i in dict_val],axis=0, weights=weights)

        fusedar = np.average([ dict_ar[i][test][0:minv] for i in dict_ar],axis=0, weights=weights)

        fusedvallist.append(fusedval)
        fusedarlist.append(fusedar)
        for i in range(len(fusedval)):
            f.write(str(fusedval[i])+' '+str(fusedar[i])+os.linesep)
    f.close()
    return fusedvallist, fusedarlist


# First copy the files to be fused to  fusedir
# Then run 
dict_val,dict_ar = read_valence_arousal()

#write to new file, give a distinctive name with a fuse method
mean_fuse(dict_val,dict_ar,filename='fused.txt')

#write to new file
#mean_weighted_fuse(dict_val,dict_ar)




