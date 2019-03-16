import tensorflow as tf
import _pickle as cPickle
import numpy as np

import model
from config import train_path, train_path_i, train_path_p, test_path, test_path_i, test_path_p, valid_path, valid_path_i, valid_path_p
import config

def train(picklepath):
    """
    训练模型
    :param picklepath: 处理完的idx文件 train
    :return:
    """
    sources, targets, lens = cPickle.load(open(picklepath, "rb"))

    sources = np.array(sources)
    targets = np.array(targets)
    lens = np.array(lens)

    ### 打乱顺序
    shuffle_idx = np.random.permutation(len(lens))
    sources = sources[shuffle_idx]
    targets = targets[shuffle_idx]
    lens = lens[shuffle_idx]

    ner = model.NER()
    ner.buildModel()

    saver = tf.train.Saver()


    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        epoches = len(lens) // config.batchsize

        for k in range(config.max_iter):
            for i in range(epoches+1):
                batch_source = sources[i*config.batchsize: (i+1)*config.batchsize]
                batch_target = targets[i*config.batchsize: (i+1)*config.batchsize]
                batch_lens = lens[i*config.batchsize: (i+1)*config.batchsize]

                if len(batch_source) > 0:
                    feed_dict = {
                        ner.sources : batch_source,
                        ner.targets : batch_target,
                        ner.lens : batch_lens,
                        ner.batchsize : len(batch_lens)
                    }


                    loss = sess.run(ner.loss, feed_dict=feed_dict)
                    print(loss)
                    sess.run(ner.train_op, feed_dict=feed_dict)
            saver.save(sess, config.checkpoints_path)


def val(picklepath):
    """
    :param picklepath: 测试文件的位置
    :return:
    """
    sources, targets, lens = cPickle.load(open(picklepath, "rb"))

    sources = np.array(sources)
    targets = np.array(targets)
    lens = np.array(lens)

    ner = model.NER()
    ner.buildModel()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ### load model
        saver.restore(sess, config.checkpoints_path)

        epoches = len(lens) // config.batchsize

        for i in range(epoches + 1):
            batch_source = sources[i * config.batchsize: (i + 1) * config.batchsize]
            batch_target = targets[i * config.batchsize: (i + 1) * config.batchsize]
            batch_lens = lens[i * config.batchsize: (i + 1) * config.batchsize]

            if len(batch_source) > 0:
                feed_dict = {
                    ner.sources: batch_source,
                    ner.targets: batch_target,
                    ner.lens: batch_lens,
                    ner.batchsize: len(batch_lens)
                }

                accuracy = sess.run(ner.accuracy, feed_dict=feed_dict)
                print(accuracy)


def predict(sentences, word2index, index2tag):
    """
    对单个句子进行标注
    :param sentences: "I like China"
    :param word2index:
    :param index2tag:
    :return:
    """

    sentences = sentences.strip().split()

    sources = [word2index.get(w, 1) for w in sentences] ###对应的ID或者UNK

    if len(sources) > config.maxlens: ### 大于最大长度
        sources = sources[: config.maxlens]
        lens = [config.maxlens]
    else:
        lens = [len(sources)]
        sources.extend([0] * (config.maxlens-len(sources)))


    sources = np.array([sources])
    lens = np.array(lens)

    ner = model.NER()
    ner.buildModel()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ### load model
        saver.restore(sess, config.checkpoints_path)

        batch_source = sources
        batch_lens = lens

        feed_dict = {
            ner.sources: batch_source,
            ner.lens: batch_lens,
            ner.batchsize: 1
        }

        decode_tags, _ = sess.run(ner.decode_op, feed_dict=feed_dict)
        tags = decode_tags[0]
        tags = [index2tag[tags[i]] for i in range(lens[0])]
        return tags



if __name__ == "__main__":

    ### load word2index, index2word, target2index, index2target

    word2index = cPickle.load(open(config.word2index_path, "rb"))
    index2word = cPickle.load(open(config.index2word_path, "rb"))
    target2index = cPickle.load(open(config.target2index_path, "rb"))
    index2target = cPickle.load(open(config.index2target_path, "rb"))

    # train(train_path_i)

    # val(valid_path_i)

    tags = predict("President Obama is speaking at the White House", word2index, index2target)
    print(tags)