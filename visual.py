from gensim.models import Word2Vec as we
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from ggplot import *
import time
from sklearn.manifold import TSNE
import random


# Load model
model = we.load("bnc_model")
#vec_samp = model.wv["-mett"].tolist()

'''
# Build Vocabulary
vocab = {}
with open("utterances_bnc.txt") as text:
	for line in text:
		if (len(line.split())==1):		#cuz utterances with only one word are eliminated
			continue
		for word in line.split():
			if word not in vocab:
				vocab[word] = 1
			else:
				vocab[word]+= 1

voc = [w for w in vocab if vocab[w]>2]		#minimum number of occurrences is set to 3

# Select verbs

verb_f = open("uniq_bnc_verbs.txt","r")
verbs = verb_f.read().splitlines()
voc = [w for w in voc if w in verbs]
'''
#Vectorization
#y = voc
#X = [model.wv[v].tolist() for v in voc]

###### trial ######
'''
ot_sim1 = ['ot', 'ket', 'phend', 'khept', 'rɨk', 'rokt']
ot_sim2 = ['rond', 'lakt', 'dipt', 'lupt', 'hot', 'keŋs', 'kuŋd', 'lukt', 'luk', 'ris']
ot_sim3 = ['putt', 'mas', 'ŋept', 'khokt', 'sup', 'leŋs', 'ript', 'pemd', 'kɨpt', 'thand', 'pek', 'sil', 'bhukt', 'chekt', 'hept']

ot_sim1 = ['melt']
ot_sim2 = ['break']
ot_sim3 = ['talk']
'''

ot = []		#labels
y = []		#verbs
thres = 0.65	#similarity threshold
verb_list = ['melt','break','kill','teach','hear','talk']
for verb in verb_list:
	for item in model.wv.most_similar(verb, topn=100):
		if (item[1]<thres):
			break
		elif item[0] not in y:
			y.append(item[0])
			ot.append(verb)

'''
for item in model.wv.most_similar('melt',topn=100):
	if (item[1]<thres):
		break
	elif item[0] not in y:
		y.append(item[0])
		ot.append("melt")

for item in model.wv.most_similar('break',topn=100):
	if (item[1]<thres):
		break
	elif item[0] not in y:
		y.append(item[0])
		ot.append("break")
#	else:
#		posit = [i for i,x in enumerate(y) if x == item[0]]
#		ot[posit[0]] = "fusion"

for item in model.wv.most_similar('kill',topn=100):
	if (item[1]<thres):
		break
	elif item[0] not in y:
		y.append(item[0])
		ot.append("kill")
#	else:
#		posit = [i for i,x in enumerate(y) if x == item[0]]
#		ot[posit[0]] = "fusion"

for item in model.wv.most_similar('teach',topn=100):
	if (item[1]<thres):
		break
	elif item[0] not in y:
		y.append(item[0])
		ot.append("teach")

for item in model.wv.most_similar('hear',topn=100):
	if (item[1]<thres):
		break
	elif item[0] not in y:
		y.append(item[0])
		ot.append("hear")
#	else:
#		posit = [i for i,x in enumerate(y) if x == item[0]]
#		ot[posit[0]] = "fusion"

for item in model.wv.most_similar('talk',topn=100):
	if (item[1]<thres):
		break
	elif item[0] not in y:
		y.append(item[0])
		ot.append("talk")
#	else:
#		posit = [i for i,x in enumerate(y) if x == item[0]]
#		ot[posit[0]] = "fusion"

'''

'''
# comparison random group
comp = 500
for i in range(comp):
	ran = random.choice(voc)
	if ran not in y:
		y.append(ran)
		ot.append("others")
'''

X = [model.wv[v].tolist() for v in y]

'''
ot = []
for item in y:
	if item in ot_sim1:
		ot.append("level1")
	elif item in ot_sim2:
		ot.append("level2")
	elif item in ot_sim3:
		ot.append("level3")
	else:
		ot.append("others")
'''
#ot = [item if item in ot_sim else "0" for item in y]
###################

# Data Frame
feat_cols = [ 'dim'+str(i) for i in range(np.shape(X)[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['label'] = ot

#print ('Size of the dataframe: {}'.format(df.shape))
rndperm = np.random.permutation(df.shape[0])

# PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]

# PCA for TSNE

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)

print ('Explained variation per principal component (PCA): {}'.format(np.sum(pca_50.explained_variance_ratio_)))

# TSNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
tsne_results = tsne.fit_transform(pca_result_50[range(0,df.shape[0])])

df_tsne = df.loc[range(0,df.shape[0]),:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

# PLOT TSNE
chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=1000,alpha=0.5) \
        + ggtitle("tSNE dimensions colored by digit")
print(chart)


'''
# PLOT PCA
chart = ggplot( df.loc[rndperm[:1000],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=10,alpha=0.8) \
        + ggtitle("Trial")
print(chart)
'''
