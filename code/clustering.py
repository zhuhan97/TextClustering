from time import time
import jieba
import os
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics  
import matplotlib.pyplot as plt 
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn import (manifold,datasets,decomposition,ensemble,random_projection)
import logging
import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
digital = re.compile('[0-9]*')
train_path = os.path.join(os.getcwd(), r"Text")
fname = os.path.join(os.getcwd(),r'Model',r'Doc2Vec_model')
n_neighbors = 30
kmeans_clusters = 6
testsizeperdoc=1000
def use_dbscan(x):
	db = DBSCAN(eps=0.5, min_samples=3).fit(x)
	return db.labels_
	#return db
	
def use_k_means(X):
	kmeans = KMeans(n_clusters=kmeans_clusters, random_state=0).fit(X)
	return kmeans.labels_
	#return kmeans.predict(X)
	#kmeans.cluster_centers_
	#return kmeans

def plot_embedding_2d(X, y,title=None):
    #坐标缩放到[0，1)区间
    x_min,x_max = np.min(X,axis=0),np.max(X,axis=0)
    X = (X - x_min)/(x_max - x_min)
    #降维后坐标为（X[i，0]，X[i，1]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(X.shape[0]):
        ax.text(X[i,0],X[i,1],str(y[i]),
                color = plt.cm.Set1(y[i]/10.),
                fontdict={'weight':'bold','size':9})
    if title is not None:
        plt.title(title)
    plt.show()

def dimensionality_reduce(X, y):
	logging.info("Computing Isomap embedding")
	t0 = time()
	X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
	#print("Done.")
	plot_embedding_2d(X_iso, y,
               "Isomap projection of the doc2vec (time %.2fs)" %
               (time() - t0))

def LDA_DR(X, y):
	#线形判别分析（Linear Discriminant Analysis，LDA）从64维降到2，3维
	logging.info("Computing LDA projection")
	X = np.array(X)
	X2 = X.copy()
	X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
	t0 = time()
	X_lda = lda(n_components=3).fit_transform(X2,y)
	plot_embedding_2d(X_lda[:,0:2],y,"LDA of Kmeans")
	#plot_embedding_3d(X_lda,"LDA 3D (time %.2fs)" %(time() - t0))



def get_doc_vec(doclist):
	model = Doc2Vec.load(fname)
	#vector = model.infer_vector(["system", "response"])
	doc2vecList = []
	logging.info("Inferring vectors for documents...")
	for doc in doclist:
		doc2vecList.append(model.infer_vector(doc))
	return doc2vecList
	
def read_data(file_name):
	logging.info("Downloading {}...".format(file_name))
	doc = codecs.open(file_name, encoding = 'utf-8')
	sentences = []
	for line in doc:
		tmp = line.split(" ")
		s = jieba.lcut(digital.sub('',tmp[1]).replace('\n','').replace('\r','').replace('，','').replace('。',''))
		if s:
			sentences.append(s)
		if(len(sentences) == testsizeperdoc):
			break
	doc.close()
	#logging.info("Have finished downloading {}".format(file_name))
	return sentences

def save_results(fname, results, ip):
	fileObject = open(fname, 'w')  
	for i in range(len(results)):  
		fileObject.write("{} {}\n".format(str(ip+i),str(results[i])))  
	fileObject.close() 

def main():
	train_files = os.listdir(train_path)
	sentences = []
	for f in train_files:
		sentences.extend(read_data(os.path.join(train_path,f)))
	assert(sentences != [])
	doc2vecList = get_doc_vec(sentences)
	print("--------------------")
	print("Sample size: ", len(doc2vecList))
	print("Doc2Vec size: ", len(doc2vecList[0]))
	print("--------------------")
	labels = use_k_means(doc2vecList)+1
	#labels = use_dbscan(doc2vecList)+2
	#save_results("PredictLabels/201702_pl.txt", labels,200001)
	#dimensionality_reduce(doc2vecList, labels)
	LDA_DR(doc2vecList, labels)
	
if __name__ == "__main__":
	main()
