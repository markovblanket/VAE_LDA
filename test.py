import numpy as np
import pickle
import os
import tensorflow as tf

vocab = 'data/20news_clean/vocab.pkl'
dataset_tr = 'data/20news_clean/train.txt.npy'
data_tr = np.load(dataset_tr, encoding = 'latin1')
vocab_content= np.load(vocab, encoding = 'latin1')
mydict = {'george':16,'amber':19}

# print(data_tr[0])
# for word in data_tr[0]:
# 	print(list(vocab_content.keys())[list(vocab_content.values()).index(word)])
idx=50
data_to_show=data_tr[idx]
my_str=[list(vocab_content.keys())[list(vocab_content.values()).index(word)] for word in data_to_show]	
h_dim=50
a = 1*np.ones((1 , h_dim)).astype(np.float32)
mu2 = (np.log(a).T-np.mean(np.log(a),1)).T
var2 = ( ( (1.0/a)*( 1 - (2.0/h_dim) ) ).T +
	( 1.0/(h_dim*h_dim) )*np.sum(1.0/a,1) ).T  
print('a',a)
print('mu2',mu2)
print('var2',var2)


# print(' '.join(my_str))

# print(vocab_content)

# for t in range(len(data_tr)):
# 	print(t,':',len(data_tr[t]))
# vocab = pickle.load(open(vocab,'rb'))
# h_dim=10
# a = 1*np.ones((1 , h_dim)).astype(np.float32)
# os.system("CUDA_VISIBLE_DEVICES=0 python3.5 run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 200")


