# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
import time
import os
import random
import load
import models.model as model

import tools
import sys

def main():
    s={
        'nh1':300,
        'nh2':300,
        'win':3,
        'emb_dimension':300,
        'lr':0.1,
        'lr_decay':0.5,
        'max_grad_norm':5,
        'seed':345,
        'nepochs':50,
        'batch_size':16,
        'keep_prob':1.0,
        'check_dir':'./checkpoints',
        'display_test_per':5,
        'lr_decay_per':10
    }

    
    # load the dataset
    train_set,test_set,dic,embedding=load.atisfold()
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    vocab = set(dic['words2idx'].keys())
    vocsize = len(vocab)

    test_lex,  test_y, test_z  = test_set[0:1000]

    y_nclasses = 2
    z_nclasses = 5


    with tf.Session() as sess:

        rnn = model.Model(
            nh1=s['nh1'],
            nh2=s['nh2'],
            ny=y_nclasses,
            nz=z_nclasses,
            de=s['emb_dimension'],
            cs=s['win'],
            lr=s['lr'],
            lr_decay=s['lr_decay'],
            embedding=embedding,
            max_gradient_norm=s['max_grad_norm'],
            model_cell='lstm'
        )

        checkpoint_dir = s['check_dir']
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        def dev_step(cwords):
            feed={
                rnn.input_x:cwords,
                rnn.keep_prob:1.0,
                rnn.batch_size:s['batch_size']
            }
            fetches=rnn.sz_pred
            sz_pred=sess.run(fetches=fetches,feed_dict=feed)
            return sz_pred
        print "测试结果："
        predictions_test=[]
        groundtruth_test=[]
        for batch in tl.iterate.minibatches(test_lex, test_z, batch_size=s['batch_size']):
            x, z = batch
            x = load.pad_sentences(x)
            x = tools.contextwin_2(x, s['win'])
            predictions_test.extend(dev_step(x))
            groundtruth_test.extend(z)

        res_test = tools.conlleval(predictions_test, groundtruth_test, '')

        print res_test

if __name__ == '__main__':
    main()




