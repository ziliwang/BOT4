import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import datetime
import time
import os
import crash


class SentimentCNN():

    def __init__(self,
                 filter_sizes="3,4,5",
                 num_filters=128,
                 dropout_keep_prob=0.5,
                 l2_reg_lambda=0.0,
                 batch_size=64,
                 num_epochs=200,
                 evaluate=True,
                 dev_percentage=0.1,
                 evaluate_every=200,
                 allow_soft_placement=True,
                 log_device_placement=False,
                 max_convergence_count=10
                 ):
        """
        parameter set
        show
        """
        self.__dev_percentage = dev_percentage
        self.__filter_sizes = filter_sizes
        self.__num_filters = num_filters
        self.__dropout_keep_prob = dropout_keep_prob
        self.__l2_reg_lambda = l2_reg_lambda
        self.__batch_size = batch_size
        self.__num_epochs = num_epochs
        self.__evaluate = evaluate
        self.__dev_percentage = dev_percentage
        self.__evaluate_every = evaluate_every
        self.__allow_soft_placement = allow_soft_placement
        self.__log_device_placement = log_device_placement
        self.__max_convergence_count = max_convergence_count
        self.__mask_y = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            if not isinstance(X, list):
                raise ValueError('input parameters error')
            else:
                X = np.array(X)
        shuffle_indices = np.random.permutation(np.arange(y.shape[0]))
        X = X[shuffle_indices]
        y = y[shuffle_indices]
        if self.__evaluate:
            dev_sample_index = -1 * int(self.__dev_percentage * float(len(y)))
            x_train = X[:dev_sample_index]
            x_dev = X[dev_sample_index:]
            y_train = y[:dev_sample_index]
            y_dev = y[dev_sample_index:]
            print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        else:
            x_train = X
            y_train = y
        # Training
        # ==================================================
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            session_conf = tf.ConfigProto(
                  allow_soft_placement=self.__allow_soft_placement,
                  log_device_placement=self.__log_device_placement)
            self.__sess = tf.Session(config=session_conf)
            with self.__sess.as_default():
                self.__cnn = TextCNN(
                            length=x_train.shape[1],
                            width=x_train.shape[2],
                            num_classes=y_train.shape[1],
                            filter_sizes=list(
                                    map(int, self.__filter_sizes.split(","))),
                            num_filters=self.__num_filters,
                            l2_reg_lambda=self.__l2_reg_lambda
                            )
                # Define Training procedure
                self.__optimizer = tf.train.AdamOptimizer(1e-3)
                self.__grads_and_vars = self.__optimizer.compute_gradients(
                                                            self.__cnn.loss)
                self.__global_step = tf.Variable(
                                    0, name="global_step", trainable=False)
                self.__train_op = self.__optimizer.apply_gradients(
                                        self.__grads_and_vars,
                                        global_step=self.__global_step)
                if self.__evaluate:
                    # Keep track of gradient values and sparsity (optional)
                    grad_summaries = []
                    for g, v in self.__grads_and_vars:
                        if g is not None:
                            grad_hist_summary = tf.summary.histogram(
                                    "{}/grad/hist".format(v.name), g)
                            sparsity_summary = tf.summary.scalar(
                                    "{}/grad/sparsity".format(v.name),
                                    tf.nn.zero_fraction(g))
                            grad_summaries.append(grad_hist_summary)
                            grad_summaries.append(sparsity_summary)
                    grad_summaries_merged = tf.summary.merge(
                                                            grad_summaries)
                    # Output directory for models and summaries
                    timestamp = str(int(time.time()))
                    arg_str = "_lambda_{}_filter_sizes_{}_dropout_{}_filter_num_{}".format(
                                    self.__l2_reg_lambda,
                                    self.__filter_sizes,
                                    self.__dropout_keep_prob,
                                    self.__num_filters)
                    timestamp += arg_str
                    out_dir = os.path.abspath(
                        os.path.join(os.path.curdir, "runs", timestamp))
                    self.__out_dir = out_dir
                    print("Writing to {}\n".format(out_dir))

                    # Summaries for loss and accuracy
                    loss_summary = tf.summary.scalar("loss", self.__cnn.loss)
                    acc_summary = tf.summary.scalar("score", self.__cnn.score)

                    # Train Summaries
                    self.__train_summary_op = tf.summary.merge(
                            [loss_summary, acc_summary,
                             grad_summaries_merged])
                    train_summary_dir = os.path.join(out_dir, "summaries", "train")
                    self.__train_summary_writer = tf.summary.FileWriter(
                                        train_summary_dir, self.__sess.graph)

                    # Dev summaries
                    self.__dev_summary_op = tf.summary.merge(
                                    [loss_summary, acc_summary])
                    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                    self.__dev_summary_writer = tf.summary.FileWriter(
                                    dev_summary_dir, self.__sess.graph)
                # save
                self.__saver = tf.train.Saver(tf.global_variables(),
                                              max_to_keep=1)
                # Initialize all variables
                self.__sess.run(tf.global_variables_initializer())
                # Generate batches
                batches = batch_iter(
                                    list(zip(x_train, y_train)),
                                    self.__batch_size,
                                    self.__num_epochs)
                # Training loop. For each batch...
                convergence_count = 0
                min_loss = 99999
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    self.__mask_y = y_batch
                    self.__train_step(x_batch, y_batch, show=self.__evaluate)
                    current_step = tf.train.global_step(
                                            self.__sess, self.__global_step)
                    if self.__evaluate:
                        if current_step % self.__evaluate_every == 0:
                            print("\nEvaluation:")
                            current_loss = self.__dev_step(x_dev, y_dev)
                            print("")
                            if current_loss < min_loss:
                                convergence_count = 0
                                min_loss = current_loss
                            else:
                                convergence_count += 1
                        if convergence_count > self.__max_convergence_count:
                            self.__last_step = current_step
                            break
                self.__last_step = current_step

    def decision_function(self, X):
        all_predictions = []
        all_scores = False
        predictions = self.__graph.get_operation_by_name("full_cont_layer/predictions").outputs[0]
        input_x = self.__graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = self.__graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        for x_batch in batch_iter(X, self.__batch_size, 1, shuffle=False):
                batch_predictions = self.__sess.run(
                            [predictions],
                            {input_x: x_batch,
                             dropout_keep_prob: 1.0})
                if all_predictions:
                    all_predictions = np.vstack((all_predictions, batch_predictions))
                else:
                    all_predictions = batch_predictions
        return all_predictions

    def predict(self, X):
        all_predictions, = self.decision_function(X)
        return all_predictions

    def score(self, X, y):
        all_predictions = self.predict(X)
        correct_predictions = float(sum(all_predictions == y))
        return correct_predictions/float(len(y))

    def save(self):
        print(self.__out_dir)
        _file = os.path.join(self.__out_dir, 'model')
        print(_file)
        self.__saver.save(self.__sess, _file, global_step=self.__last_step)

    def load(self, _file):
        m_file = tf.train.latest_checkpoint(_file)
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=self.__allow_soft_placement,
              log_device_placement=self.__log_device_placement)
            self.__sess = tf.Session(config=session_conf)
            saver = tf.train.import_meta_graph("{}.meta".format(m_file))
            saver.restore(self.__sess, m_file)

    def __train_step(self, x_batch, y_batch, show=True):
        feed_dict = {
          self.__cnn.input_x: x_batch,
          self.__cnn.input_y: y_batch,
          self.__cnn.dropout_keep_prob: self.__dropout_keep_prob
        }
        if not show:
            _, step, loss, accuracy = self.__sess.run(
                [self.__train_op, self.__global_step, self.__cnn.loss,
                 self.__cnn.accuracy],
                feed_dict)
        else:
            _, step, summaries, loss, accuracy, pred = self.__sess.run(
                [self.__train_op, self.__global_step, self.__train_summary_op,
                 self.__cnn.loss, self.__cnn.score, self.__cnn.predictions],
                feed_dict)
            BOTscore = BOTscoreCal(pred, y_batch)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, BOTscore {}".format(
                        time_str, step, loss, accuracy, BOTscore))
            botsummary = tf.Summary(value=[
                tf.Summary.Value(tag="BOTscore", simple_value=BOTscore),
                ])
            self.__train_summary_writer.add_summary(botsummary, step)
            self.__train_summary_writer.add_summary(summaries, step)

    def __dev_step(self, x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          self.__cnn.input_x: x_batch,
          self.__cnn.input_y: y_batch,
          self.__cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy, pred = self.__sess.run(
            [self.__global_step, self.__dev_summary_op, self.__cnn.loss,
             self.__cnn.score, self.__cnn.predictions],
            feed_dict)
        BOTscore = BOTscoreCal(pred, y_batch)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}, BOTscore {}".format(
                    time_str, step, loss, accuracy, BOTscore))
        botsummary = tf.Summary(value=[
            tf.Summary.Value(tag="BOTscore", simple_value=BOTscore),
            ])
        self.__dev_summary_writer.add_summary(botsummary, step)
        self.__dev_summary_writer.add_summary(summaries, step)
        return loss


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and
    softmax layer.
    """
    def __init__(self, length, width, num_classes, filter_sizes,
                 num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32,
                                      [None, length, width],
                                      name="input_x")
        self.x = tf.reshape(self.input_x, [-1, length, width, 1])
        self.input_y = tf.placeholder(tf.float32,
                                      [None, num_classes],
                                      name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, width, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                conv = tf.nn.conv2d(
                    self.x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("full_cont_layer"):
            # W = tf.Variable(tf.truncated_normal([num_filters_total, class_num], stddev=0.1), name="W")
            W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.predictions = tf.nn.xw_plus_b(self.h_drop, W, b, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score,
                                                            #  labels=self.input_y)
            losses = tf.sqrt(tf.square(self.predictions - self.input_y))
            """sparse softmax cross entropy do not need to transform labels to
            one-hot matrix. and """
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("score"):
            self.score = tf.reduce_mean(tf.sqrt(tf.square(self.predictions - self.input_y)), name="score")


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def expect_margin(predictions, answer):
    predict_sign = np.sign(predictions)
    answer_sign = np.sign(answer)
    margin_array = []
    for m in range(answer.shape[0]):
        row = []
        for n in range(answer.shape[1]):
            a = answer[m, n]
            p = predictions[m, n]
            p_s = predict_sign[m, n]
            a_s = answer_sign[m, n]
            if p_s == a_s:
                row.append(min(abs(a), abs(p)))
            elif p_s != a_s and a*p != 0:
                row.append(-1*(abs(a)+abs(p)))
            else:
                row.append(-1*a)
        margin_array.append(row)
    margin_array = np.array(margin_array)
    return np.sum(margin_array,0)


def BOTscoreCal(pred, ans):
    ans = np.array(list(ans))
    margin1 = expect_margin(pred, np.array(ans))  # 选手预测的三日预期收益
    margin2 = expect_margin(np.array(ans), np.array(ans))  # 完美预测的三日预期收益
    margin_rate = np.divide(margin1, margin2)
    BOTscore = np.sum(margin_rate)
    return BOTscore
