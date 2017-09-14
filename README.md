# Keyphrase Extraction
Source codes of our EMNLP2016 paper [Keyphrase Extraction Using Deep Recurrent Neural Networks on Twitter](http://jkx.fudan.edu.cn/~qzhang/paper/keyphrase.emnlp2016.pdf)

## Preparation
You need to prepare  the pre-trained word vectors.
* Pre-trained word vectors. Download [GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/)


## Details
Joint RNN model

* data文件夹存储数据集

* checkpoints文件夹存储模型训练得到的参数

* main.py是主程序

* models/model.py定义了joint-rnn模型

* models/bi_lstm_model.py 用双向lstm代替rnn

* load.py用于加载数据集

* tools.py定义了一些工具函数

## Requirement
tensorflow0.11 + tensorlayer

