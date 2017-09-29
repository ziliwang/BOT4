import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
from ipdb import set_trace


class TextLSTM(object):
    """
    A DNN+LSTM classifier for NLU. stripping Embedding.
    structure: input layer + LSTM layer + mean_pool layer + softmax layer
    """
    def __init__(self, embedding_size=300, lstm_layer_num=3,
                 max_time_size=50, cell_size=100, forget_bias=0.0,
                 l2_reg_lambda=0.0, class_num=8):
        """
        define the lstm structure.
        arguments:
            embedding_size: word embedding size[300].
            max_time_size: max times in dnn, for nlp, this mean the max words in
                a document[50].
            lstm_layer_num: multiple lstm hidden layer number, this lstm allow
                learn more sophisticated conditional distributions[3].
            cell_size: lstm cell size. the cell bigger size, the more information
                cell remenber[100].
            l2_reg_lambda: the lambda of l2 regularization in softmax layer[0.0].
        """
        # begin
        """
        constant store in model. benefit: when load model can show the constant
        arguments.
        dropout not used in test step, move to outside.
        """
        _l2_reg_lambda = tf.constant(l2_reg_lambda, dtype=tf.float32,
                                     name="l2_reg_lambda")
        _lstm_layer_num = tf.constant(lstm_layer_num, dtype=tf.int32,
                                      name="lstm_layer_num")
        _cell_size = tf.constant(cell_size, dtype=tf.int32,
                                 name="cell_size")
        _max_time_size = tf.constant(max_time_size, dtype=tf.int32,
                                     name="max_time_size")
        """
        Placeholders for input, output and dropout.
        """
        # inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
        #                     dtype=tf.float32)
        self.input_x = tf.placeholder(
                shape=(None, embedding_size, max_time_size),
                dtype=tf.float32,
                name="input_x")
        batch_size = tf.shape(self.input_x)[0]
        self.input_y = tf.placeholder(shape=(None, class_num), dtype=tf.float32,
                                      name="input_y")
        self.input_keep_prob = tf.placeholder(tf.float32,
                                              name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(
                        tf.float32,
                        name="output_keep_prob"
                            )
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        def lstm_cell_func():
            # LSTM Cell, hidden size larger, remenber more detail
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                    cell_size,
                    forget_bias=forget_bias,
                    state_is_tuple=True)
            """
            add dropout, dnn dropout different from cnn.
            in_keep_prob: input keep probability(the probability of h_t == 0).
            out_keep_prob: output keep probability(the probability of h_{t+1} == 0).
            """

            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                        lstm_cell,
                        input_keep_prob=self.input_keep_prob,
                        output_keep_prob=self.output_keep_prob)
            """What's the benefit of multiple LSTM hidden layer?
            point 1: An interesting property of multilayer LSTMs is that it allows to
                perform hierarchical processing on difficult temporal tasks, and more
                naturally capture the structure of sequences.
            point 2: The purpose of using multilayer RNN cells is to learn more
                sophisticated conditional distributions"""
            return lstm_cell
        cell = tf.nn.rnn_cell.MultiRNNCell(
                        [lstm_cell_func() for _ in range(lstm_layer_num)], state_is_tuple=True)
        with tf.name_scope("lstm"):
            state = cell.zero_state(batch_size, tf.float32)  # sents counte
        # with tf.name_scope("lstm"):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for time_step in range(max_time_size):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (h_t, state) = cell(self.input_x[:,:,time_step], state)
                h = h_t
        # 全连阶层
        with tf.name_scope("full_cont_layer"):
            W1 = tf.Variable(tf.truncated_normal([cell_size, class_num], stddev=0.1), name="W1")
            W2 = tf.Variable(tf.truncated_normal([cell_size, class_num], stddev=0.1), name="W2")
            W3 = tf.Variable(tf.truncated_normal([cell_size, class_num], stddev=0.1), name="W3")
            b1 = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b1")
            b2 = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b2")
            b3 = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b3")
            l2_loss += tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
            l2_loss += tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(b3)
            self.scores = tf.nn.xw_plus_b(h, W1, b1, name="scores")
            # self.score = tf.matmul(h, W) + b
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                            #  labels=self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores+1e-10, labels=self.input_y)
            """sparse softmax cross entropy do not need to transform labels to
            one-hot matrix. and """
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                        tf.cast(correct_predictions, "float"), name="accuracy")
