import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
# import matplotlib.pyplot as plt
import pickle

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def log_dir_init(fan_in, fan_out,topics=50):
    return tf.log((1.0/topics)*tf.ones([fan_in, fan_out]))

tf.reset_default_graph()
class VAE(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.002, batch_size=200):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_count=0
        self.z_batch_norm_flag=network_architecture['z_batch_flag']
        self.beta_batch_norm_flag=network_architecture['beta_batch_flag']        
        self.phi_batch_norm_flag=network_architecture['phi_batch_flag']        
	self.prob=float(network_architecture['keep_prob'])
        '''----------------Inputs----------------'''
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.keep_prob = tf.placeholder(tf.float32)
#        self.keep_prob = float(network_architecture['keep_prob'])

        '''-------Constructing Laplace Approximation to Dirichlet Prior--------------'''
        self.h_dim = float(network_architecture["n_z"])
        self.h_dim = int(network_architecture["n_z"])
       # print('h_dim: ',self.h_dim)
        self.a = 1*np.ones((1 , self.h_dim)).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),1)).T)
        self.var2 = tf.constant(  ( ( (1.0/self.a)*( 1 - (2.0/self.h_dim) ) ).T +
                                ( 1.0/(self.h_dim*self.h_dim) )*np.sum(1.0/self.a,1) ).T  )

        # Create autoencoder network
        self._create_network()
        self._create_loss_optimizer()
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights(**self.network_architecture)

        self.z_mean,self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"],
                                      self.network_weights["biases_recog"])

        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((1, n_z), 0, 1,
                               dtype=tf.float32)
        # self.z = tf.add(self.z_mean,
        #                 tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.sigma = tf.exp(self.z_log_sigma_sq)
        self.x_reconstr_mean = \
            self._generator_network(self.z,self.network_weights["weights_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,
                            n_input, n_z,keep_prob,z_batch_flag,beta_batch_flag,phi_batch_flag):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.get_variable('h1',[n_input, n_hidden_recog_1]),
            'h2': tf.get_variable('h2',[n_hidden_recog_1, n_hidden_recog_2]),
            'out_mean': tf.get_variable('out_mean',[n_hidden_recog_2, n_z]),
            'phi1': tf.get_variable('phi1',[n_input,5]),                        
            # 'phi2': tf.Variable(tf.zeros([n_input,n_input,n_z], dtype=tf.float32)),            
            'phi2': tf.get_variable('phi2',[5,n_input,n_z]),
            'out_log_sigma': tf.get_variable('out_log_sigma',[n_hidden_recog_2, n_z])}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'phi1': tf.Variable(tf.zeros([5], dtype=tf.float32)),
            'phi2': tf.Variable(tf.zeros([n_input,n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h2': tf.Variable(xavier_init(n_z, n_hidden_gener_1))}
        return all_weights

    def _recognition_network(self, weights, biases):
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        # layer_3 = self.transfer_fct(tf.add(tf.tensordot(self.x, weights['phi'],axes=((1),(0))),
        #                                    biases['phi'])) 
        layer_3 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['phi1']),
                                           biases['phi1']))

        layer_4 = self.transfer_fct(tf.add(tf.tensordot(layer_3, weights['phi2'],axes=((1),(0))),
                                           biases['phi2'])) 
        #print('layer_4_shape',layer_4.get_shape())

        if (self.phi_batch_norm_flag):
            self.phi=tf.nn.softmax(tf.contrib.layers.batch_norm(layer_4))
            print ('-'*20)            
            print('batch_norm for Phi Enabled')
            print ('-'*20)
        else :
            self.phi=tf.nn.softmax(layer_4)
            print ('-'*20)            
            print('batch_norm for Phi Disabled')
            print ('-'*20)


        #print('new_phi',self.phi.get_shape())
        layer_do = tf.nn.dropout(layer_2, self.keep_prob)
        if (self.z_batch_norm_flag):
            z_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_mean']),
                            biases['out_mean']))
            z_log_sigma_sq =tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_log_sigma']),biases['out_log_sigma']))
            print ('-'*20)            
            print('batch_norm for z Enabled')
            print ('-'*20)
        else:
            z_mean = tf.add(tf.matmul(layer_do, weights['out_mean']),biases['out_mean'])
            z_log_sigma_sq =tf.add(tf.matmul(layer_do, weights['out_log_sigma']),biases['out_log_sigma'])                        
            print ('-'*20)            
            print('batch_norm for z Disabled')
            print ('-'*20)
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self,z, weights):
        self.layer_do_0 = tf.nn.dropout(tf.nn.softmax(z), self.keep_prob)
        # '''sampled theta'''
        # self.theta_l=self.layer_do_0
        if (self.beta_batch_norm_flag):
            self.beta=tf.nn.softmax(tf.contrib.layers.batch_norm(weights['h2']))       
            print ('-'*20)
            print('batch_norm for beta Enabled')
            print ('-'*20)            
        else:
            self.beta=tf.nn.softmax(weights['h2'])       
            print ('-'*20)
            print('batch_norm for beta Disabled')
            print ('-'*20)

        x_reconstr_mean = tf.add(tf.matmul(self.layer_do_0, tf.nn.softmax(weights['h2'])),0.0)
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        self.x_reconstr_mean+=1e-10
        self.phi+=1e-10
        ''' E_{q(theta|w;mu_0,Sigma_0)q(z|w;phi)}[log p(z|theta)]'''        
        theta_expand=tf.expand_dims(self.layer_do_0,-1)+1e-10
        t_z_p_loss=tf.reduce_sum(tf.matmul(self.phi,tf.log(theta_expand)),[1,2])   
       # print ('t_z_p_loss',t_z_p_loss.get_shape())        

        ''' E_q(z|w;phi)[log q(z|w;phi)]'''
        z_z_q_loss=tf.reduce_sum(self.phi*tf.log(self.phi),[1,2])
        #print ('z_z_q_loss',z_z_q_loss.get_shape())        

        ''' E_{q(theta,z|w)}{log p(w|z,theta)}'''
        # reconstr_loss = -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean),1)#/tf.reduce_sum(self.x,1)
        # print ('reconstr_loss_original',reconstr_loss.get_shape())        
        # recons_loss=tf.reduce_sum(tf.multiply(self.x,tf.matrix_diag_part(tf.tensordot(self.phi,tf.log(self.beta),axes=((2),(0))))))
        recons_loss=tf.reduce_sum(tf.multiply(self.phi,tf.transpose(tf.log(self.beta))),2)
        recons_loss=tf.reduce_sum(self.x*recons_loss,1)
        #print ('reconstr_loss_mine',recons_loss.get_shape())
        ''' E_{q(theta|w)}{log (p(theta|w)/q(theta|w))}'''             
        latent_loss = 0.5*( tf.reduce_sum(tf.div(self.sigma,self.var2),1)+\
        tf.reduce_sum( tf.multiply(tf.div((self.mu2 - self.z_mean),self.var2),
                  (self.mu2 - self.z_mean)),1) - self.h_dim +\
                           tf.reduce_sum(tf.log(self.var2),1)  - tf.reduce_sum(self.z_log_sigma_sq  ,1) )
       # print ('latent_loss',latent_loss.get_shape())
        # reconstr_loss = \
        #     -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean),1)#/tf.reduce_sum(self.x,1)
            

        # self.cost = tf.reduce_mean(reconstr_loss) + tf.reduce_mean(latent_loss) # average over batch
        # self.cost = tf.reduce_mean(-self.recons_loss-self.t_z_p_loss+self.z_z_q_loss) + tf.reduce_mean(latent_loss) # average over batch

        self.cost = tf.reduce_mean(-recons_loss-t_z_p_loss+z_z_q_loss+latent_loss) # average over batch
       # self.cost=tf.reduce_mean(latent_loss)

        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.99).minimize(self.cost)

    def partial_fit(self, X):                
        opt, cost,emb = self.sess.run((self.optimizer, self.cost,self.network_weights['weights_gener']['h2']),feed_dict={self.x: X,self.keep_prob:self.prob})
        # print('trace_monitor',trace_monitor)        
        # self.sess.run((self.layer_3_print),feed_dict={self.x: X,self.keep_prob: .75})
        return cost,emb

    def test(self, X):
        cost = self.sess.run((self.cost),feed_dict={self.x: np.expand_dims(X, axis=0),self.keep_prob:1.0})
        return cost
    def topic_prop(self, X):
        """heta_ is the topic proportion vector. Apply softmax transformation to it before use.
        """
        theta_ = self.sess.run((self.z),feed_dict={self.x: np.expand_dims(X, axis=0),self.keep_prob: 1.0})
        return theta_
