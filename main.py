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
        'nepochs':150,
        'batch_size':16,
        'keep_prob':0.5,
        'check_dir':'./checkpoints',
        'display_test_per':3,
        'lr_decay_per':10
    }

    train_set,test_set,dic,embedding=load.atisfold()
    
    
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    train_lex, train_y, train_z = train_set

    tr = int(len(train_lex)*0.9)
    valid_lex, valid_y, valid_z = train_lex[tr:], train_y[tr:], train_z[tr:]
    train_lex, train_y, train_z = train_lex[:tr], train_y[:tr], train_z[:tr]
    test_lex,  test_y, test_z  = test_set

    print 'len(train_data) {}'.format(len(train_lex))
    print 'len(valid_data) {}'.format(len(valid_lex))
    print 'len(test_data) {}'.format(len(test_lex))

    vocab = set(dic['words2idx'].keys())
    vocsize = len(vocab)
    print 'len(vocab) {}'.format(vocsize)
    print "Train started!"

    y_nclasses = 2
    z_nclasses = 5

    nsentences = len(train_lex)


    with tf.Session() as sess:
        
        rnn=model.Model(
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

        checkpoint_dir=s['check_dir']
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_prefix=os.path.join(checkpoint_dir,'model')

        def train_step(cwords,label_y,label_z):
            feed={
                rnn.input_x:cwords,
                rnn.input_y:label_y,
                rnn.input_z:label_z,
                rnn.keep_prob:s['keep_prob'],
                rnn.batch_size:s['batch_size']
            }
            fetches=[rnn.loss,rnn.train_op]
            loss,_=sess.run(fetches=fetches,feed_dict=feed)
            return loss

        def dev_step(cwords):
            feed={
                rnn.input_x:cwords,
                rnn.keep_prob:1.0,
                rnn.batch_size:s['batch_size']
            }
            fetches=rnn.sz_pred
            sz_pred=sess.run(fetches=fetches,feed_dict=feed)
            return sz_pred

        saver=tf.train.Saver(tf.all_variables())
        sess.run(tf.initialize_all_variables())

        best_f=-1
        best_e=0
        test_best_f=-1
        test_best_e=0
        best_res=None
        test_best_res=None
        for e in xrange(s['nepochs']):
            tools.shuffle([train_lex,train_y,train_z],s['seed'])
            t_start=time.time()
            for step,batch in enumerate(tl.iterate.minibatches(train_lex,zip(train_y,train_z),batch_size=s['batch_size'])):
                input_x,target=batch
                label_y,label_z=zip(*target)
                input_x=load.pad_sentences(input_x)
                label_y=load.pad_sentences(label_y)
                label_z=load.pad_sentences(label_z)
                cwords=tools.contextwin_2(input_x,s['win'])
                loss=train_step(cwords,label_y,label_z)

                print 'loss %.2f' % loss,' [learning] epoch %i>> %2.2f%%' % (e,s['batch_size']*step*100./nsentences),'completed in %.2f (sec) <<\r' % (time.time()-t_start),

                sys.stdout.flush()

            #VALID

            predictions_valid=[]
            predictions_test=[]
            groundtruth_valid=[]
            groundtruth_test=[]
            for batch in  tl.iterate.minibatches(valid_lex,valid_z,batch_size=s['batch_size']):
                x,z=batch
                x=load.pad_sentences(x)
                x=tools.contextwin_2(x,s['win'])
                predictions_valid.extend(dev_step(x))
                groundtruth_valid.extend(z)

            res_valid=tools.conlleval(predictions_valid,groundtruth_valid,'')

            if res_valid['f']>best_f:
                best_f=res_valid['f']
                best_e=e
                best_res=res_valid
                print '\nVALID new best:',res_valid
                path = saver.save(sess=sess, save_path=checkpoint_prefix, global_step=e)
                print "Save model checkpoint to {}".format(path)
            else:
                print '\nVALID new curr:',res_valid

            #TEST
            if e%s['display_test_per']==0:
                for batch in tl.iterate.minibatches(test_lex, test_z, batch_size=s['batch_size']):
                    x,z = batch
                    x = load.pad_sentences(x)
                    x = tools.contextwin_2(x, s['win'])
                    predictions_test.extend(dev_step(x))
                    groundtruth_test.extend(z)


                res_test = tools.conlleval(predictions_test, groundtruth_test, '')

                if res_test['f'] > test_best_f:
                    test_best_f = res_test['f']
                    test_best_e=e
                    test_best_res=res_test
                    print 'TEST new best:',res_test
                else:
                    print 'TEST new curr:',res_test

            # learning rate decay if no improvement in 10 epochs
            if e-best_e>s['lr_decay_per']:
                sess.run(fetches=rnn.learning_rate_decay_op)
            lr=sess.run(fetches=rnn.lr)
            print 'learning rate:%f' % lr
            if lr<1e-5:break
            print

        print "Train finished!"
        print 'Valid Best Result: epoch %d:  ' % (best_e),best_res
        print 'Test Best Result: epoch %d:  ' %(test_best_e),test_best_res

if __name__ == '__main__':
    main()
