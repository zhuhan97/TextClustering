# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:51:29 2017

@author: zhuhan
"""

import numpy as np
import multiprocessing
import os
from gensim.corpora.dictionary import Dictionary
from gensim.models.word2vec import Word2Vec
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.wrappers  import TimeDistributed, Bidirectional
from keras.preprocessing import sequence
from keras.layers.core import RepeatVector, Activation, Dense
from keras.layers.recurrent import LSTM	
from keras.models import load_model, Model
from keras.layers import Input
from collections import Counter
from keras import layers
import sys
import logging
import time
import re
import codecs
import jieba

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sys.path.append("D:\Anaconda3\workspace\Chatbot")
sys.path.append("/home/zhuh/ChatBot")
from attention_decoder import AttentionDecoder

digital = re.compile('[0-9]*')

w2v_dim = 128
maxlen = 100
n_iterations = 1 
n_exposures = 1
window_size = 7
input_length = 100
cpu_count = multiprocessing.cpu_count()

c_dim = 50	#the dim of context
decode_dim = 50   #the size of decoder's hidden layer
seq2seq_batch_size = 32
seq2seq_epochs = 5
loop = 5

w2v_file = os.path.join(os.getcwd(),r"Model/word2vec_model")
seq2seq_file = os.path.join(os.getcwd(),r"Model/seq2seq_model.h5")
seq2seq_weights_file = os.path.join(os.getcwd(),r'Model/seq2seq_model_weights.h5')
train_path = os.path.join(os.getcwd(), r"test")


new_seq2seq = False
new_w2v = False

#读取文件并进行分词
def read_data(file_name):
	logging.info("Start downloading {}...".format(file_name))
	doc = codecs.open(file_name, encoding = 'utf-8')
	sentences = []
	for line in doc:
		tmp = line.split(" ")
		s = jieba.lcut(digital.sub('',tmp[1]).replace('\n','').replace('\r','').replace('，','').replace('。',''))
		if s:
			sentences.append(s)
	doc.close()
	logging.info("Have finished downloading {}".format(file_name))
	return sentences

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model = None, sentences = None):
	if (sentences is not None) and (model is not None):
		gensim_dict = Dictionary()
		gensim_dict.doc2bow(model.wv.vocab.keys(),
							allow_update=True)
		w2indx = {v: k+1 for k, v in gensim_dict.items()}
		w2vec = {word: model[word] for word in w2indx.keys()}
		idx2w = {k+1: v for k, v in gensim_dict.items()}
		def parse_dataset(sentences):
			data=[]
			for sentence in sentences:
				new_txt = []
				for word in sentence:
					try:
						new_txt.append(w2indx[word])
					except:
						new_txt.append(0)
				data.append(new_txt)
			return data
			
		sentences=parse_dataset(sentences)
		sentences= sequence.pad_sequences(sentences, maxlen=maxlen)
		return w2indx, w2vec, idx2w, sentences
	else:
		logging.info ('No data provided...')
	
def word2vec_model(sentences, new_model = True):
	if (not new_model) and os.path.exists(w2v_file):
		logging.info("Loading word2vec_model from: ", w2v_file)
		model = Word2Vec.load(w2v_file)
	else:
		logging.info("Building word2vec_model...")
		model = Word2Vec(size = w2v_dim,
				min_count = n_exposures,
				window = window_size,
				workers = cpu_count)
		model.build_vocab(sentences)
		model.train(sentences, epochs=n_iterations, total_examples=model.corpus_count)
		model.save(w2v_file)
	return model

def seq2seq_model(n_symbols, embedding_weights, x_train, y_train, x_test, y_test, idx_to_word, new_model = False):
	if (not new_model) and os.path.exists(seq2seq_file):
		logging.info("Reading seq2seq_model from {}".format(seq2seq_file))
		en_de_model = load_model(seq2seq_file, custom_objects={'AttentionDecoder':AttentionDecoder})
	else:
		logging.info("Building new seq2seq_model...")
		inputs = Input(shape=(maxlen,))  
		out = Embedding(input_dim=n_symbols,   
							output_dim=w2v_dim,   
							input_length=maxlen,
							mask_zero = True,
							weights = [embedding_weights],
							trainable = True, name="Embedding_1")(inputs)   
		out = Bidirectional(LSTM(c_dim,return_sequences = True), merge_mode = 'sum')(out)
		out = AttentionDecoder(decode_dim, n_symbols)(out)
		out = Dense(n_symbols, activation="relu", name="Dense_1")(out)
		#en_de_model.add(RepeatVector(maxlen))
		#en_de_model.add(TimeDistributed(Dense(maxlen, activation="linear")))   
		out = Activation('softmax', name = "Activation_1")(out)
		en_de_model = Model(inputs = inputs, outputs=out)
		layer = en_de_model.get_layer(name = "Embedding_1")
		print(layer.input_shape,"  ", layer.output_shape)
		logging.info('Compiling...')   
		time_start = time.time()   
		en_de_model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])   
		time_end = time.time()   
		logging.info('Cost time of compilation is:%fsecond!' % (time_end - time_start)) 
	
	logging.info('Start training...')
	for iter_num in range(loop):   
		en_de_model.fit(x_train, y_train, batch_size=seq2seq_batch_size, epochs = seq2seq_epochs, verbose = 2) 
		out_predicts = en_de_model.predict(x_test)   
		for i_idx, out_predict in enumerate(out_predicts):   
			predict_sequence = []   
			ground_truth = []
			for predict_vector in out_predict:   
				next_index = np.argmax(predict_vector)   
				if next_index != 0:
					next_token = idx_to_word[next_index]  
					predict_sequence.append(next_token)
				next_index = np.where(y_test[i_idx] == True)
				if next_index[0].shape[0] != 0:
					next_token = idx_to_word[next_index[0][0]]
					ground_truth.append(next_token)
			print('Target output:', str.join(ground_truth))   
			print('Predict output:', str.join(predict_sequence)) 
		logging.info('Current iter_num is:%d' % iter_num)  
		en_de_model.save(seq2seq_file)
		en_de_model.save_weights(seq2seq_weights_file) 
	return en_de_model

def get_in_output(index_dict, word_vectors, x, y):
	n_symbols = len(index_dict) + 1 
	embedding_weights = np.zeros((n_symbols, w2v_dim))
	y = np.zeros((len(x), maxlen, n_symbols), dtype=bool) 
	for s_index, tar_tmp in enumerate(x):
		for t_index, token in enumerate(tar_tmp): 
			try:
				y[s_index, t_index, word_to_idx[token]] = 1
			except:
				continue
	for word, index in index_dict.items():
		embedding_weights[index, :] = word_vectors[word]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.001)
	print("x_train.shape: ", x_train.shape)
	print("y_train.shape: ", y_train.shape)
	return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

#def draw_result():

def main():
	train_files = os.listdir(train_path)
	sentences = []
	for f in train_files:
		sentences.extend(read_data(os.path.join(train_path,f)))
	assert(sentences != [])
	w2v_model = word2vec_model(sentences, new_w2v)
	bow, w2v, idx2w, sentences = create_dictionaries(w2v_model, sentences)
	n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_in_output(bow, w2v, sentences, sentences)
	del bow, w2v
	print('-----------------------------')   
	print('Vocab size:', n_symbols, 'unique words')   
	print('Input max length:', maxlen, 'words')   
	print('Dimension of hidden vectors:', w2v_dim)   
	print('Number of samples:', len(sentences))   
	print('-----------------------------')   
	en_de_model = seq2seq_model(n_symbols, embedding_weights, x_train, y_train, x_test, y_test, idx2w, new_seq2seq)


	
if __name__ == '__main__':   
	main()  	


