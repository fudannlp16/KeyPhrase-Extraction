# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Model(object):

    def __init__(self,
                 nh1,
                 nh2,
                 ny,
                 nz,
                 de,
                 cs,
                 lr,
                 lr_decay,
                 embedding,
                 max_gradient_norm,
                 model_cell='rnn',
                 model='basic_model',
                 nonstatic=False):

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=None)
        self.input_x=tf.placeholder(tf.int32,shape=[None,None,cs],name='input_x')
        self.input_y=tf.placeholder(tf.int32,shape=[None,None],name="input_y")
        self.input_z=tf.placeholder(tf.int32,shape=[None,None],name='input_z')
        self.keep_prob=tf.placeholder(dtype=tf.float32,name='keep_prob')

        self.lr=tf.Variable(lr,dtype=tf.float32)

        self.learning_rate_decay_op = self.lr.assign(
            self.lr * lr_decay)


        #Creating embedding input
        with tf.device("/cpu:0"),tf.name_scope('embedding'):
            if nonstatic:
                W=tf.constant(embedding,name='embW',dtype=tf.float32)
            else:
                W=tf.Variable(embedding,name='embW',dtype=tf.float32)
            inputs=tf.nn.embedding_lookup(W,self.input_x)
            inputs=tf.reshape(inputs,[self.batch_size,-1,cs*de])

        #Droupout embedding input
        inputs=tf.nn.dropout(inputs,keep_prob=self.keep_prob,name='drop_inputs')

        #Create the internal multi-layer cell for rnn
        if model_cell=='rnn':
            single_cell0=tf.nn.rnn_cell.BasicRNNCell(nh1)
            single_cell1=tf.nn.rnn_cell.BasicRNNCell(nh1)
            single_cell2=tf.nn.rnn_cell.BasicRNNCell(nh2)
        elif model_cell=='lstm':
            single_cell0=tf.nn.rnn_cell.BasicLSTMCell(nh1,state_is_tuple=True)
            single_cell1=tf.nn.rnn_cell.BasicLSTMCell(nh1,state_is_tuple=True)
            single_cell2=tf.nn.rnn_cell.BasicLSTMCell(nh2,state_is_tuple=True)
        elif model_cell=='gru':
            single_cell0=tf.nn.rnn_cell.GRUCell(nh1)
            single_cell1=tf.nn.rnn_cell.GRUCell(nh1)
            single_cell2=tf.nn.rnn_cell.GRUCell(nh2)
        else:
            raise 'model_cell error!'
        #DropoutWrapper rnn_cell
        single_cell0 = tf.nn.rnn_cell.DropoutWrapper(single_cell0,output_keep_prob=self.keep_prob)
        single_cell1 = tf.nn.rnn_cell.DropoutWrapper(single_cell1, output_keep_prob=self.keep_prob)
        single_cell2 = tf.nn.rnn_cell.DropoutWrapper(single_cell2, output_keep_prob=self.keep_prob)
        
        self.init_state=single_cell1.zero_state(self.batch_size,dtype=tf.float32) 
        
        
        #Bi-RNN1

        x_len = tf.cast(tf.shape(inputs)[1], tf.int64)
        batch=2
        with tf.variable_scope('bi_rnn1'):
            self.outputs1,self.state1=tf.nn.bidirectional_dynamic_rnn(
                single_cell0,
                single_cell1,
                inputs,
                sequence_length=[x_len]*batch,
                dtype=tf.float32
            )

        self.outputs1=tf.concat(2,self.outputs1)

        
        #RNN2
        with tf.variable_scope('rnn2'):
            self.outputs2,self.state2=tf.nn.dynamic_rnn(
                cell=single_cell2,
                inputs=self.outputs1,
                initial_state=self.init_state,
                dtype=tf.float32
            )

        #outputs_y
        with tf.variable_scope('output_sy'):
            w_y=tf.get_variable("softmax_w_y",[2*nh1,ny])
            b_y=tf.get_variable("softmax_b_y",[ny])
            outputs1=tf.reshape(self.outputs1,[-1,2*nh1])
            sy=tf.nn.xw_plus_b(outputs1,w_y,b_y)
            self.sy_pred = tf.reshape(tf.argmax(sy, 1), [self.batch_size, -1])
        #outputs_z
        with tf.variable_scope('output_sz'):
            w_z = tf.get_variable("softmax_w_z", [nh2, nz])
            b_z = tf.get_variable("softmax_b_z", [nz])
            outputs2 = tf.reshape(self.outputs2, [-1, nh2])
            sz = tf.nn.xw_plus_b(outputs2, w_z,b_z)
            self.sz_pred = tf.reshape(tf.argmax(sz, 1), [self.batch_size, -1])
        #loss
        with tf.variable_scope('loss'):
            label_y = tf.reshape(self.input_y, [-1])
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(sy, label_y)
            label_z = tf.reshape(self.input_z, [-1])
            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(sz, label_z)
            self.loss=tf.reduce_sum(0.5*loss1+0.5*loss2)/tf.cast(self.batch_size,tf.float32)

        tvars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,tvars),max_gradient_norm)
        optimizer=tf.train.GradientDescentOptimizer(self.lr)
        self.train_op=optimizer.apply_gradients(zip(grads,tvars))

    def cost(output, target):
        # Compute cross entropy for each frame.
        cross_entropy = target * tf.log(output)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(cross_entropy)