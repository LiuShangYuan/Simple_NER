maxlens = 40

batchsize = 100

num_tags = 10

embedding_dim = 50

vocabulary_size = 23625

max_iter = 100 ### 训练的最大迭代次数

checkpoints_path = "./train/checkpoints"

glove_path = "./data/glove.6B.50d.txt" ### glove词向量的位置

#### 原始
train_path = "./data/train.txt"
test_path = "./data/test.txt"
valid_path = "./data/valid.txt"

### 处理过
train_path_p = "./data/train.pickle"
test_path_p = "./data/test.pickle"
valid_path_p = "./data/valid.pickle"

### 处理过后转成idx的形式
train_path_i = "./data/train_i.pickle"
test_path_i = "./data/test_i.pickle"
valid_path_i = "./data/valid_i.pickle"


### word2index, index2word, target2index, index2target

word2index_path = "./data/vocab/word2index.pickle"
index2word_path = "./data/vocab/index2word.pickle"
target2index_path = "./data/vocab/target2index.pickle"
index2target_path = "./data/vocab/index2target.pickle"


