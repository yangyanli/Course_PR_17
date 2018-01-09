# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from mxnet import nd
from mxnet.gluon import nn,Parameter
from mxnet import init
from mxnet import cpu
from mxnet.gluon.loss import Loss, L2Loss,  _apply_weighting

class PrimaryConv(nn.Block):
    def __init__(self,dim_vector,n_channels,kernel_size,padding,context=cpu,strides=(1,1),**kwargs):
        super(PrimaryConv, self).__init__(**kwargs)
        self.dim_vector = dim_vector
        self.n_channels= n_channels
        self.batch_size = 0
        self.capsules_index = ['dim_'+str(i) for i in range(n_channels)]
        for idx in self.capsules_index:
            setattr(self, idx, nn.Conv2D(channels=dim_vector, 
                    kernel_size=kernel_size, strides=strides,
                    padding=padding,activation='relu'))
    def concat_outputs(self,conv_list,axis):
        concat_vec = conv_list[0]
        concat_vec = nd.reshape(concat_vec,shape=(self.batch_size, self.dim_vector,-1))
        for i in range(1, len(conv_list)):
            concat_vec = nd.concat(concat_vec, nd.reshape(conv_list[i],shape=(self.batch_size, self.dim_vector,-1)), dim=axis)
        return concat_vec   
    def forward(self, x):
        self.batch_size = x.shape[0]
        conv_list = [getattr(self,idx)(x).expand_dims(axis=-1) for idx in self.capsules_index]    
        outputs = self.concat_outputs(conv_list,axis=2)
        assert outputs.shape == (self.batch_size, 8, 1152)
        # v_primary = self.squash(outputs,axis=1)
        # assert outputs.shape == (self.batch_size, 8, 1152)
        return outputs


class DigitCaps(nn.Block):
    def __init__(self,num_capsule,dim_vector,context=cpu,iter_routing=3,**kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule #10
        self.dim_vector = dim_vector #16
        
        self.iter_routing = iter_routing #3

        self.batch_size = 1 
        self.input_num_capsule = 1152
        self.input_dim_vector = 8
        self.context = context

        #  (1, 1152, 10, 8, 16)
        self.W_ij  = self.params.get(
            'weight',shape=(
                1,
                self.input_num_capsule,
                self.num_capsule,
                self.input_dim_vector,
                self.dim_vector
                )) 

    def squash(self,vectors,axis):
        epsilon = 1e-9
        vectors_l2norm = nd.square(vectors).sum(axis=axis,keepdims=True)# 1,10,1,1
        scale_factor = vectors_l2norm / (1 + vectors_l2norm) 
        vectors_squashed = scale_factor * (vectors / nd.sqrt(vectors_l2norm+epsilon)) # element-wise
        return vectors_squashed

  
    def forward(self, x):

        self.batch_size, self.input_dim_vector, self.input_num_capsule = x.shape#batch_size,8,1152
        x_exp = x.expand_dims(axis=1)
        x_exp = x_exp.expand_dims(axis=4)#batch_size,1,8,1152,1
        x_tile = x_exp.tile(reps=[1, self.num_capsule, 1, 1, 1])#batch_size,10,8,1152,1
        x_trans = x_tile.transpose(axes=(0,3,1,2,4))#batch_size,1152,10,8,1
        W = self.W_ij.data().tile(reps=[self.batch_size,1,1,1,1])


        x_dot = x_trans.reshape(shape=(-1,self.input_dim_vector,1))#(8,1)
        W_dot = W.reshape(shape=(-1,self.input_dim_vector,self.dim_vector))#(8,16)
        u_hat = nd.batch_dot(W_dot,x_dot,transpose_a=True)
        u_hat = u_hat.reshape(shape=(self.batch_size,self.input_num_capsule,self.num_capsule,self.dim_vector,-1))
        b_IJ = nd.zeros((self.batch_size, self.input_num_capsule,self.num_capsule,1,1),ctx=self.context)
        u_hat_stopped = nd.stop_gradient(u_hat, name='stop_gradient')


        for r_iter in range(self.iter_routing):
            c_IJ = nd.softmax(b_IJ, axis=2)
            s_J = nd.multiply(c_IJ, u_hat)
            s_J = s_J.sum(axis=1,keepdims=True)#bs,1,10,16,1
            # print('s_J',s_J[0,0,0])
            v_J = self.squash(s_J,axis=3)#bs,1,10,16,1
            v_J_tiled = v_J.tile(reps=[1, 1152, 1, 1, 1])

            if self.iter_routing > 1:
                u_produce_v = nd.stop_gradient(nd.multiply(u_hat_stopped, v_J_tiled),transpose_a=True)
                u_produce_v = u_produce_v.reshape(shape=(self.batch_size, self.input_num_capsule, self.num_capsule, 1, 1))#bs,1152,10,1,1
                b_IJ = nd.stop_gradient(b_IJ+u_produce_v, name ="update_b_IJ" )
        return v_J
       


class Length(nn.Block):
    def __init__(self, **kwargs):
        super(Length, self).__init__(**kwargs)

    def forward(self, x):
        #(batch_size, 1, 10, 16, 1) =>(batch_size,10, 16)=> (batch_size, 10, 1)
        x_shape = x.shape
        x = x.reshape(shape=(x_shape[0],x_shape[2],x_shape[3]))
        x_l2norm = nd.sqrt((x.square()).sum(axis=-1))
        return x_l2norm #batch_size,10,1


