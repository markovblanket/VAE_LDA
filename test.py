import numpy as np
import pickle
import os
import tensorflow as tf


def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

vocab = 'data/20news_clean/vocab.pkl'
dataset_tr = 'data/20news_clean/train.txt.npy'
data_tr = np.load(dataset_tr, encoding = 'latin1')
vocab_content= np.load(vocab, encoding = 'latin1')
mydict = {'george':16,'amber':19}
doc_len=[len(data_tr[k]) for k in range(10)]
vocab_size=len(vocab_content)
data_tr=data_tr[0:200]
#data_tr = np.array([np.diag(onehot(doc.astype('int'),vocab_size)) for doc in data_tr if np.sum(doc)!=0])
data_tr = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
non_zero_vec=np.array([np.count_nonzero(data_tr[k,:]) for k in range(len(data_tr))])
print('non_zero_vec',non_zero_vec)
print ('min is ',np.min(non_zero_vec),' max is ', np.max(non_zero_vec))
print(np.count_nonzero(data_tr[0,:]))
print('data_tr',data_tr.shape)
#print(data_tr[0])
A=np.array([1,2,3])
B=np.array([[1,2,1],[2,3,5],[4,5,6]])
D=np.reshape(B,[9,])
print D

vec_1=tf.constant([[[1,1,1],[2,2,2]],[[3,3,3],[4,5,1]]], dtype=tf.float32)
vec_2=tf.constant([[2,1,1]],dtype=tf.float32)
vec_3=tf.constant([[3,2,4]],dtype=tf.float32)
diag_mat=np.diag(data_tr)
print('diag_mat_shape',diag_mat.shape)
C=np.empty([200,1995,1995])
#for k in range(200):
#    C[k]=np.diag(data_tr[k])
#print ('C_size is', C.shape)
# print('A',A)
# print('B',B)
# print ('multiply',np.multiply(A,np.diag(B)))
# C=np.multiply(A,np.diag(B))
# print ('output',np.multiply(A,np.diag(B)))
# print ('sum',np.sum(C))

# C=np.tensordot(A,B,axes=((1),(0)))
# print('A',A)
# print('B',B[0,:,:])
# print('C',C[0,:,:])
# print(np.trace(A))
# print(data_tr.shape)
# print(len(np.bincount(data_tr[6], minlength=len(vocab_content))))

# print(data_tr[0])
# for word in data_tr[0]:
# 	print(list(vocab_content.keys())[list(vocab_content.values()).index(word)])
# idx=50
# data_to_show=data_tr[idx]
# my_str=[list(vocab_content.keys())[list(vocab_content.values()).index(word)] for word in data_to_show]	
# h_dim=50
# a = 1*np.ones((1 , h_dim)).astype(np.float32)
# mu2 = (np.log(a).T-np.mean(np.log(a),1)).T
# var2 = ( ( (1.0/a)*( 1 - (2.0/h_dim) ) ).T +
# 	( 1.0/(h_dim*h_dim) )*np.sum(1.0/a,1) ).T  
# print('a',a)
# print('mu2',mu2)
# print('var2',var2)


# print(' '.join(my_str))

# print(vocab_content)

# for t in range(len(data_tr)):
# 	print(t,':',len(data_tr[t]))
# vocab = pickle.load(open(vocab,'rb'))
# h_dim=10
# a = 1*np.ones((1 , h_dim)).astype(np.float32)
# os.system("CUDA_VISIBLE_DEVICES=0 python3.5 run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 200")


