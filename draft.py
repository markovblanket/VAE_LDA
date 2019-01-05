import numpy as np
import pickle

array=[1,2,1,3,4,4,5,1]

dataset_tr = 'data/20news_clean/train.txt.npy'
data_tr = np.load(dataset_tr, encoding = 'latin1')
dataset_te = 'data/20news_clean/test.txt.npy'
data_te = np.load(dataset_te, encoding = 'latin1')
vocab = 'data/20news_clean/vocab.pkl'
vocab = pickle.load(open(vocab,'rb'))
vocab_size=len(vocab)

