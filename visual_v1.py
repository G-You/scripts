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
model = we.load("bnc_model_w3")

ot = []		#labels
thres = 0.6	#similarity threshold
verb_list = ['melt','break','kill','see','learn','change','love']
#verb_list = ['melt','change','die']
y = []
#y = verb_list		#verbs
random.shuffle(verb_list)

for verb in verb_list:
	for item in model.wv.most_similar(verb, topn=100):
		if (item[1]<thres):
			break
		elif item[0] not in y:
			y.append(item[0])
			ot.append(verb)

# Vectorization
X = [model.wv[v].tolist() for v in y]

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
# K means trial
features = np.array(X)
kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
#for i in range(len(X)):
#	print(y[i],kmeans.labels_[i])
'''


# PLOT TSNE
chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=1000,alpha=0.5) \
        + ggtitle("tSNE dimensions colored by digit")
print(chart)
