文件结构
original_data：
    trnTweet
    testTweet
    GoogleNews-vectors-negative300.txt
data_process.py
data_set.pkl
embedding.pkl
main_example.py



1.original_data为原始数据目录
trnTweet和testTweet是处理好的原始tweet
trainTweet 78760
testTweet 33755
每一行包含tweet和对应的hashtag,tweet和hashtag以\t隔开
GoogleNews-vectors-negative300.txt为用Google news预训练的300维词向量(available at https://code.google.com/archive/p/word2vec/)


2.data_process.py 为数据处理脚本
运行python data_process.py会生成处理好的数据
data_set.pkl和embedding.pkl


3.data_set.pkl 数据文件
data_set=[train_set,test_set,dicts]
train_set=[train_lex,train_y,train_z]
test_set=[test_lex,test_y,test_z]
dicts = {'words2idx': words2idx, 'labels2idx': labels2idx}(labels2idx好像用不到)

words2idx,labelsidx数据类型为字典,key代表单词,value代表对应的id(数字)
train_lex,train_y,train_z test_lex,test_y,test_z是处理好的数据(id化)
train_lex为tweet.train_y表示tweet每个单词对应的标记,是hashtag对应1,不是hashtag对应0,trian_z表示带有位置的标记,0表示不是hashtag,1表示hashtag的开始位置,2表示hashtag的中间位置,3表示hangtag的结尾位置,4表示对应的hashtag就是一个单词.即labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
train_lex,train_y,train_z test_lex,test_y,test_z的数据类型为python嵌套列表
如train_lex[0]表示第1行tweet,也是一个列表,里面存贮着单词对应的id(数字)


4.embedding.pkl用GoogleNews-vectors-negative300预训练的300维词向量,包含数据集中所有单词的词向量,padding对应的id为0,全0初始化
使用方法
embedding=cPickle.load(open('embedding.pkl'))


5.main_example.py为使用示例


