import tensorflow as tf
import tensorflow.contrib.slim as slim

import config



class NER(object):
    def __init__(self):
        pass


    def buildModel(self):
        ###Placeholder
        self.sources = tf.placeholder(tf.int32, [None, config.maxlens], name="source")
        self.targets = tf.placeholder(tf.int32, [None, config.maxlens], name="target")
        self.lens = tf.placeholder(tf.int32, [None], "lens")
        self.batchsize = tf.placeholder(tf.int32, [], name="batchsize")



        self.embedding = tf.Variable(tf.truncated_normal([config.vocabulary_size, config.embedding_dim]), dtype=tf.float32)

        embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.sources)

        ### Bi-LSTM
        cell_fw = tf.keras.layers.LSTMCell(100)
        cell_bw = tf.keras.layers.LSTMCell(100)

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                   embedding_inputs,
                                                                   sequence_length=self.lens,
                                                                   dtype=tf.float32)


        ### 拼接双向LSTM的输出
        output_fw, output_bw = outputs
        output = tf.concat([output_fw, output_bw], axis=-1)
        ### 将LSTM的输出映射到n_tags
        self.W = tf.get_variable("W", [2 * 100, config.num_tags])

        matricized_output = tf.reshape(output, [-1, 2 * 100])
        matricized_unary_scores = tf.matmul(matricized_output , self.W)

        self.unary_scores = tf.reshape(matricized_unary_scores, [self.batchsize, config.maxlens, config.num_tags])

        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.unary_scores, self.targets, self.lens)
        self.loss = tf.reduce_mean(-log_likelihood)


        self.optimizer = tf.train.AdamOptimizer(0.001)

        self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)

        ### 解码
        self.decode_op = tf.contrib.crf.crf_decode(self.unary_scores, self.transition_params, self.lens)

        decode_tags, _ = tf.contrib.crf.crf_decode(self.unary_scores, self.transition_params, self.lens)

        ### [batch_size, max_len]
        mask = tf.sequence_mask(self.lens, maxlen=config.maxlens, dtype=tf.float32)
        res = tf.equal(self.targets, decode_tags)
        out = tf.cast(res, tf.float32)
        mask_out = out * mask
        self.accuracy = tf.reduce_sum(mask_out) / tf.cast(tf.reduce_sum(self.lens), dtype=tf.float32)