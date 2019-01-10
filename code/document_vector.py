import numpy as np
import codecs
import re
import gensim
import os
import collections
import smart_open
import random
import jieba

# Set file names for train and test data
data_dir = os.path.join(os.getcwd(), r'Text')
#lee_train_file = data_dir + os.sep + '201701.txt'
#lee_test_file = test_data_dir + os.sep + 'lee.cor'

def simple_preprocess(doc, max_len=60):
	digital = re.compile('[0-9]*')
	doc = doc.split(" ")
	s = jieba.lcut(digital.sub('', doc[1]).replace('\n','').replace('\r','').replace('，','').replace('。',''))
	if len(s) > max_len:
		return s[:max_len]
	return s

#gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)
#Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long
def read_corpus(dirname, tokens_only=False):
    files = os.listdir(dirname)
    i=-1
    for fname in files:
        print("Processing ",fname)
        with codecs.open(os.path.join(dirname,fname), encoding = 'utf-8') as f:
        #with smart_open.smart_open(fname, encoding="utf-8") as f:
            for line in f:
                if tokens_only:
                    yield simple_preprocess(line)
                else:
                    i = i + 1
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(simple_preprocess(line), [i])
				
train_corpus = list(read_corpus(data_dir))
#test_corpus = list(read_corpus(lee_test_file, tokens_only=True))


#train_corpus = list(read_corpus(lee_train_file))

print(train_corpus[:2])
print(len(train_corpus))
print("--------------------------------")
model = gensim.models.doc2vec.Doc2Vec(vector_size=256, min_count=2, epochs=60)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
#from gensim.test.utils import get_tmpfile
#fname = get_tmpfile("my_doc2vec_model")
fname = os.path.join(os.getcwd(),r'Model',r'Doc2Vec_model')
model.save(fname)
#model = Doc2Vec.load(fname)  # you can continue training with the loaded model!
