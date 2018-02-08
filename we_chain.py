from gensim.models import Word2Vec as we
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from ggplot import *
import time
from sklearn.manifold import TSNE
import random
from sklearn.cluster import KMeans


# Load model
model = we.load("./models/cbow_3")

# Select verbs
verb_f = open("uniq_bnc_verbs.txt","r")
voc = verb_f.read().splitlines()
#voc = [w for w in voc if w in verbs]



#ot = []		#labels
#thres = 0.6	#similarity threshold
#verb_list = ['melt','break','kill','see','learn','change','love']
#verb_list = ['melt','change','die']
#y = []

#y = verb_list		#verbs
#random.shuffle(verb_list)
start = "melt"
chain = [start]

def form_chain(verb):
	flag = 0
	for item in model.wv.most_similar(verb, topn=2):
		if (item[0] not in chain) and (item[0] in voc) and (item[0][-1]!="'"):
			chain.append(item[0])
			flag = 1
			form_chain(item[0])
#			break
	if (flag==0):
		return

form_chain(start)		#699
#form_chain("kill")		#+34
#form_chain("freeze")	#+18



file = open("causatives_SKIP","w+")
for v in chain:
	file.write(v+"\n")

'''
for verb in verb_list:
	for item in model.wv.most_similar(verb, topn=100):
		if (item[1]<thres):
			break
		elif item[0] not in y:
			y.append(item[0])
			ot.append(verb)
'''
'''
# Vectorization
X = [model.wv[v].tolist() for v in y]

# Data Frame
feat_cols = [ 'dim'+str(i) for i in range(np.shape(X)[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['label'] = ot

#print ('Size of the dataframe: {}'.format(df.shape))
rndperm = np.random.permutation(df.shape[0])
'''
'''
# PCA for TSNE

pca_50 = PCA(n_components=100)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)

print ('Explained variation per principal component (PCA): {}'.format(np.sum(pca_50.explained_variance_ratio_)))

# TSNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
tsne_results = tsne.fit_transform(pca_result_50[range(0,df.shape[0])])

df_tsne = df.loc[range(0,df.shape[0]),:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]
'''
'''
# K means trial
features = np.array(X)
kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
#for i in range(len(X)):
#	print(y[i],kmeans.labels_[i])
'''

'''
# PLOT TSNE
chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=1000,alpha=0.5) \
        + ggtitle("tSNE dimensions colored by digit")
print(chart)
'''