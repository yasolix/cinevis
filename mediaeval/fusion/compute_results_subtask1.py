import os
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import pearsonr

testset = ['MEDIAEVAL17_00','MEDIAEVAL17_01','MEDIAEVAL17_02','MEDIAEVAL17_03','MEDIAEVAL17_04','MEDIAEVAL17_05','MEDIAEVAL17_06','MEDIAEVAL17_07','MEDIAEVAL17_08','MEDIAEVAL17_09','MEDIAEVAL17_10','MEDIAEVAL17_11','MEDIAEVAL17_12','MEDIAEVAL17_13' ]

dir_results = "teams/"
dir_gt = "ground_truth/"

names_gt = []
valence_gt = []
arousal_gt = []

for test in testset:

	names_gt.append(test)

	valence = []
	arousal = []
	
	with open('ground_truth/'+test+'_Valence-Arousal.txt','r') as f:
		f.readline() # Skip first line
		for line in f:
			lineParts = line.split()
			valence.append(float(lineParts[2]))
			arousal.append(float(lineParts[3]))

	valence_gt.append(valence)
	arousal_gt.append(arousal)

results_name = []
results_valence_mse = []
results_arousal_mse = []
results_valence_pearson = []
results_arousal_pearson = []


files = os.listdir(dir_results)
for fname in files:
	if fname.find('.txt') >=0:
		print(fname)

		with open(dir_results+fname,'r') as f:
			lines = f.read().splitlines()

		names = []
		valence_mse = []
		arousal_mse = []
		valence_pearson = []
		arousal_pearson = []

		indexes = []

		for i in range(len(lines)):
			if lines[i].find("MEDIAEVAL17") >= 0:
				indexes.append(i)
				names.append(lines[i])

		for i in range(len(names)):

			ibeg = indexes[i]+1
			if i < len(names)-1:
				iend = indexes[i+1]-1
			else:
				iend = len(lines)-1

			valence = []
			arousal = []

			for j in range(ibeg,iend+1):
				items = lines[j].split()
				valence.append(float(items[0]))
				arousal.append(float(items[1]))

			ind_gt = names_gt.index(names[i])

			minv = min(len(valence),len(valence_gt[ind_gt]))
			
			valence_true = valence_gt[ind_gt][0:minv]
			valence_predict = valence[0:minv]

			arousal_true = arousal_gt[ind_gt][0:minv]
			arousal_predict = arousal[0:minv]

			valence_mse.append(mean_squared_error(valence_true, valence_predict))
			arousal_mse.append(mean_squared_error(arousal_true, arousal_predict))
			valence_pearson.append(pearsonr(valence_true, valence_predict)[0])
			arousal_pearson.append(pearsonr(arousal_true, arousal_predict)[0])

		results_name.append(fname)
		results_valence_mse.append(sum(valence_mse)/len(valence_mse))
		results_arousal_mse.append(sum(arousal_mse)/len(arousal_mse))
		results_valence_pearson.append(sum(valence_pearson)/len(valence_pearson))
		results_arousal_pearson.append(sum(arousal_pearson)/len(arousal_pearson))


with open('results_subtask1_valence.txt','w') as f:
	f.write("run\tvalence_mse\tvalence_r\n")
	for i in range(len(results_name)):
		f.write(results_name[i]+"\t"+str(results_valence_mse[i])+"\t"+str(results_valence_pearson[i])+"\n")

with open('results_subtask1_arousal.txt','w') as f:
	f.write("run\tarousal_mse\tarousal_r\n")
	for i in range(len(results_name)):
		f.write(results_name[i]+"\t"+str(results_arousal_mse[i])+"\t"+str(results_arousal_pearson[i])+"\n")




