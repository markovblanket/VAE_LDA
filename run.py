#!/usr/bin/python
import scipy.io as sio
import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
# import matplotlib.pyplot as plt
import pickle
import sys, getopt
from models import prodlda, nvlda
'''-----------Data--------------'''
def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)
mat_address='/home/rezaee1/ferraro_user/rezaee1/VAE_LDA/output_runs/'
dataset_tr = '/home/rezaee1/ferraro_user/rezaee1/VAE_LDA/data/20news_clean/train.txt.npy'
data_tr = np.load(dataset_tr, encoding = 'latin1')
np.random.shuffle(data_tr)
data_valid=np.array([data_tr[k] for k in range(0,500)])
data_tr=np.array([data_tr[k] for k in range(500,len(data_tr))])

dataset_te = '/home/rezaee1/ferraro_user/rezaee1/VAE_LDA/data/20news_clean/test.txt.npy'
data_te = np.load(dataset_te, encoding = 'latin1')
vocab = '/home/rezaee1/ferraro_user/rezaee1/VAE_LDA/data/20news_clean/vocab.pkl'
vocab = pickle.load(open(vocab,'rb'))
vocab_size=len(vocab)
#--------------convert to one-hot representation------------------
print ('Converting data to one-hot representation')
data_tr = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
data_te = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
data_valid = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_valid if np.sum(doc)!=0])
#--------------print the data dimentions--------------------------
print ('Data Loaded')
print ('Dim Training Data',data_tr.shape)
print ('Dim Test Data',data_te.shape)
print ('Dim Dev Data',data_valid.shape)
'''-----------------------------'''

'''--------------Global Params---------------'''
n_samples_tr = data_tr.shape[0]
n_samples_te = data_te.shape[0]
docs_tr = data_tr
docs_te = data_te
docs_valid=data_valid
#batch_size=200
#learning_rate=0.002
network_architecture = \
    dict(n_hidden_recog_1=100, # 1st layer encoder neurons
         n_hidden_recog_2=100, # 2nd layer encoder neurons
         n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
         n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
         n_z=50)  # dimensionality of latent space

'''-----------------------------'''

'''--------------Netowrk Architecture and settings---------------'''
def make_network(layer1=100,layer2=100,num_topics=50,bs=200,eta=0.002,keeping_prob=0.75,z_batch_norm_flag=1,beta_batch_norm_flag=1,phi_batch_norm_flag=1):
    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1, # 1st layer encoder neurons
             n_hidden_recog_2=layer2, # 2nd layer encoder neurons
             n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
             n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
             n_z=num_topics,
             keep_prob=keeping_prob,
             z_batch_flag=z_batch_norm_flag,
             beta_batch_flag=beta_batch_norm_flag,
             phi_batch_flag=phi_batch_norm_flag
             )  # dimensionality of latent space
    batch_size=bs
    learning_rate=eta
    return network_architecture,batch_size,learning_rate



'''--------------Methods--------------'''
def create_minibatch(data,batch_size):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]



def calcPerpValid(model):
    cost=[]
    for doc in docs_valid:
        doc=doc.astype('float32')
        n_d=np.sum(doc)
        c=model.test(doc)
        cost.append(c/n_d)
        ppx=np.exp(np.mean(np.array(cost)))
    return ppx


def train(network_architecture, minibatches, type='nvlda',learning_rate=0.001,
          batch_size=200, training_epochs=100, display_step=5,iter_count=1):
    name_train="train_"+"iter_"+str(iter_count)+'_k_'+str(network_architecture['n_z'])+ \
	'_z_'+str(network_architecture['z_batch_flag'])+'_q_'+str(network_architecture['beta_batch_flag'])+'_c_'+str(network_architecture['phi_batch_flag'])
    name_valid="valid_"+"iter_"+str(iter_count)+'_k_'+str(network_architecture['n_z'])+ \
	'_z_'+str(network_architecture['z_batch_flag'])+'_q_'+str(network_architecture['beta_batch_flag'])+'_c_'+str(network_architecture['phi_batch_flag'])      
    name_beta="beta_"+"iter_"+str(iter_count)+'_k_'+str(network_architecture['n_z'])+ \
        '_z_'+str(network_architecture['z_batch_flag'])+'_q_'+str(network_architecture['beta_batch_flag'])+'_c_'+str(network_architecture['phi_batch_flag'])

#print('name is',name)
    tf.reset_default_graph()
    vae=''
    cost_array_train=np.array([])
    ppx_array_valid=np.array([])
    if type=='prodlda':
        vae = prodlda.VAE(network_architecture,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size)
    elif type=='nvlda':
        vae = nvlda.VAE(network_architecture,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size)
    emb=0
    # Training cycle
    print('training_epochs',training_epochs)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples_tr / batch_size)
        vae.epoch_count+=1
#	print('batch_size',batch_size)
        if vae.epoch_count%50==0:
          vae.learning_rate*=0.8
        # Loop over all batches
        for i in range(total_batch):
            # batch_xs = minibatches.next()
            batch_xs = next(minibatches)
#            print('batch_xs_shape:',batch_xs.shape)
            # Fit training using batch data
            cost,emb = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples_tr * batch_size

            if np.isnan(avg_cost):
                print (epoch,i,np.sum(batch_xs,1).astype(np.int),batch_xs.shape)
                print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()
#	cost_array=np.append(cost_array,avg_cost)
	#print('cost_array',cost_array)
        # Display logs per epoch step
        if epoch % display_step == 0:
           cost_array_train=np.append(cost_array_train,avg_cost)
	   valid_ppx=calcPerpValid(vae)
	   ppx_array_valid=np.append(ppx_array_valid,valid_ppx) 
           print ("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost),'learning_rate:',vae.learning_rate,'valid_ppx:',valid_ppx)
#           print('cost_array',cost_array)
    sio.savemat(mat_address+name_train,{'cost':cost_array_train})
    sio.savemat(mat_address+name_valid,{'ppx':ppx_array_valid})
    return vae,emb,name_beta

def print_top_words(beta, feature_names, n_top_words=10,name_beta=" "):
   # name_beta="beta_"+"iter_"+str(iter_count)+'_k_'+str(network_architecture['n_z'])+ \
   #     '_z_'+str(network_architecture['z_batch_flag'])+'_q_'+str(network_architecture['beta_batch_flag'])+'_c_'+str(network_architecture['phi_batch_flag'])
    beta_list=[]
    print ('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        beta_list.append(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
        print(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print ('---------------End of Topics------------------')
    with open(mat_address+name_beta+'.pkl','wb') as fp:
	pickle.dump(beta_list,fp)

def calcPerp(model,name_test):
    cost=[]
    for doc in docs_te:
        doc = doc.astype('float32')
        n_d = np.sum(doc)
        c=model.test(doc)
        cost.append(c/n_d)
    print ('The approximated perplexity is: ',(np.exp(np.mean(np.array(cost)))))
    test_ppx=np.exp(np.mean(np.array(cost)))
    sio.savemat(mat_address+name_test,{'ppx':test_ppx})


# m = 'nvlda'
# f = 100
# s = 100
# t = 5
# b = 200
# r = 1e-4
# e = 300

# minibatches = create_minibatch(docs_tr.astype('float32'))
# network_architecture,batch_size,learning_rate=make_network(f,s,t,b,r)
# print (network_architecture)

# vae,emb = train(network_architecture, minibatches,m, training_epochs=e,batch_size=batch_size,learning_rate=learning_rate)
# print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])
# calcPerp(vae)    

def main(argv):
    m = ''
    f = ''
    s = ''
    t = ''
    b = ''
    r = ''
    k = ''
    e = ''
    try:
      opts, args = getopt.getopt(argv,"hpnm:f:s:t:b:r:k:z:q:c:i:,e:",["default=","model=","layer1=","layer2=","num_topics=","batch_size=","keep_prob=","learning_rate=","z_batch_norm_flag","training_epochs"])
    except getopt.GetoptError:
        print ('CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1] -e <training_epochs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1]> -e <training_epochs>')
            sys.exit()
        elif opt == '-p':
            print ('Running with the Default settings for prodLDA...')
            print ('CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 100')
            m='prodlda'
            f=100
            s=100
            t=50
            b=200
            r=0.002
            e=100
        elif opt == '-n':
            print ('Running with the Default settings for NVLDA...')
            print ('CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300')
            m='nvlda'
            f=100
            s=100
            t=50
            b=200
            r=0.01
            e=300
        elif opt == "-m":
            m=arg
        elif opt == "-f":
            f=int(arg)
        elif opt == "-s":
            s=int(arg)
        elif opt == "-t":
            t=int(arg)
        elif opt == "-b":
            b=int(arg)
        elif opt == "-r":
            r=float(arg)
        elif opt == "-k":
            k=float(arg)
        elif opt == "-z":
            z=int(arg)                        
        elif opt == "-q":
            q=int(arg)                
        elif opt == "-c":
            c=int(arg)                
        elif opt == "-e":
            e=int(arg)
	elif opt == "-i":
	    i=int(arg)
    #minibatches = create_minibatch(docs_tr.astype('float32'))
#    print('minibatches',minibatches.shape)
    network_architecture,batch_size,learning_rate=make_network(layer1=f,layer2=s,num_topics=t,bs=b,eta=r,keeping_prob=k,z_batch_norm_flag=z,beta_batch_norm_flag=q,phi_batch_norm_flag=c)
    print (network_architecture)
    minibatches = create_minibatch(docs_tr.astype('float32'),batch_size=batch_size)
    # print (opts)
    vae,emb,name_beta = train(network_architecture, minibatches,m, training_epochs=e,batch_size=batch_size,learning_rate=learning_rate,iter_count=i)
    print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0],name_beta=name_beta)
    name_test="test_"+"iter_"+str(i)+'_k_'+str(t)+ \
        '_z_'+str(z)+'_q_'+str(q)+'_c_'+str(c)   
    calcPerp(vae,name_test)

if __name__ == "__main__":
   main(sys.argv[1:])
