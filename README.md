Implementation Keyphrase Extraction on Twitter using Tensorflow

Joint RNN model

data文件夹存储数据集

checkpoints文件夹存储模型训练得到的参数

main.py是主程序

models/model.py定义了joint-rnn模型

models/bi_lstm_model.py 用双向lstm代替rnn

load.py用于加载数据集

tools.py定义了一些工具函数

环境
tensorflow0.11 + tensorlayer

