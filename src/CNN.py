
import tensorflow as tf
import time
from datetime import datetime, timedelta
from util import get_data_4d, get_log


class DataHolder:
    """
    Class to store all the data information
    """
    def __init__(self,
                 train_dataset,
                 train_labels,
                 valid_dataset,
                 valid_labels,
                 test_dataset,
                 test_labels):
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.test_dataset = test_dataset
        self.test_labels = test_labels


class Config():
    """
    Holds model hyperparams and data information.
    The config class is used to store various hyperparameters.
    """
    def __init__(self,
                 batch_size=140,
                 patch_size=6,
                 image_size=28,
                 num_labels=10,
                 num_channels=1,
                 num_filters_1=16,
                 num_filters_2=32,
                 hidden_nodes_1=60,
                 hidden_nodes_2=40,
                 hidden_nodes_3=20,
                 learning_rate=0.9,
                 steps_for_decay=100,
                 decay_rate=0.96):
        """
        :type batch_size: int
        :type patch_size: int
        :type image_size: int
        :type num_channels: int
        :type num_filters_1: int
        :type num_filters_2: int
        :type hidden_nodes_1: int
        :type hidden_nodes_2: int
        :type hidden_nodes_3: int
        :type learning_rate: float
        :type steps_for_decay: float
        :type decay_rate: float
        """
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_nodes_1 = hidden_nodes_1
        self.hidden_nodes_2 = hidden_nodes_2
        self.hidden_nodes_3 = hidden_nodes_3
        self.learning_rate = learning_rate
        self.steps_for_decay = steps_for_decay
        self.decay_rate = decay_rate


def init_weights_bias(shape, name):
        Winit = tf.truncated_normal(shape, stddev=0.1)
        binit = tf.zeros(shape[-1])
        layer = {}
        layer["weights"] = tf.get_variable(name + "/weights",
                                           dtype=tf.float32,
                                           initializer=Winit)
        layer["bias"] = tf.get_variable(name + "/bias",
                                        dtype=tf.float32,
                                        initializer=binit)
        return layer


def apply_conv(input_tensor, layer):
    """
    Create conv layer
    """

    weights = layer['weights']
    bias = layer['bias']

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    conv_layer = tf.nn.conv2d(input=input_tensor,
                              filter=weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')

    # Add the bias to the results of the convolution.
    # A bias-value is added to each filter-channel.
    conv_layer += bias
    return conv_layer


def apply_pooling(input_layer):
    # This is 2x2 max-pooling, which means that we
    # consider 2x2 windows and select the largest value
    # in each window. Then we move 2 pixels to the next window.
    pool_layer = tf.nn.max_pool(value=input_layer,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
    pool_layer = tf.nn.relu(pool_layer)
    return pool_layer


def linear_activation(input_tensor, layer):
    """
    Method to computing linear activation
    """
    return tf.add(tf.matmul(input_tensor, layer['weights']),
                  layer['bias'])


def sgd_train(loss,
              starter_learning_rate,
              steps_for_decay,
              decay_rate):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               steps_for_decay,
                                               decay_rate,
                                               staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer.minimize(loss, global_step=global_step)


class CNNModel:

    def __init__(self, config, dataholder):
            """
            init
            """
            self.config = config
            self.valid_dataset = dataholder.valid_dataset
            self.test_dataset = dataholder.test_dataset
            self.batch_size = self.config.batch_size
            self.patch_size = self.config.patch_size
            self.image_size = self.config.image_size
            self.num_labels = self.config.num_labels
            self.num_channels = self.config.num_channels
            self.num_filters_1 = self.config.num_filters_1
            self.num_filters_2 = self.config.num_filters_2
            self.hidden_nodes_1 = self.config.hidden_nodes_1
            self.hidden_nodes_2 = self.config.hidden_nodes_2
            self.hidden_nodes_3 = self.config.hidden_nodes_3
            self.learning_rate = self.config.learning_rate
            self.steps_for_decay = self.config.steps_for_decay
            self.decay_rate = self.config.decay_rate
            self.build_graph()

    def create_placeholders(self):
        with tf.name_scope("Feed"):
            shape_train_input = (self.batch_size,
                                 self.image_size,
                                 self.image_size,
                                 self.num_channels)
            shape_train_labels = (self.batch_size,
                                  self.num_labels)
            self.train_input = tf.placeholder(tf.float32,
                                              shape=shape_train_input,
                                              name="train_input")
            self.train_labels = tf.placeholder(tf.float32,
                                               shape=shape_train_labels,
                                               name="train_labels")

    def create_logits(self):
        with tf.name_scope('Convolution_1'):
            shape1 = [self.patch_size,
                      self.patch_size,
                      self.num_channels,
                      self.num_filters_1]
            self.conv_layer_1_wb = init_weights_bias(shape1, 'Convolution_1')
            conv_layer1 = apply_conv(self.train_input,
                                     self.conv_layer_1_wb)
        with tf.name_scope('Max_pooling1'):
            pool_layer1 = apply_pooling(conv_layer1)
        with tf.name_scope('Convolution_2'):
            shape2 = [self.patch_size,
                      self.patch_size,
                      self.num_filters_1,
                      self.num_filters_2]
            self.conv_layer_2_wb = init_weights_bias(shape2, 'Convolution_2')
            conv_layer2 = apply_conv(pool_layer1,
                                     self.conv_layer_2_wb)
        with tf.name_scope('Max_pooling2'):
            pool_layer2 = apply_pooling(conv_layer2)
        with tf.name_scope('Reshape'):
            shape = pool_layer2.get_shape().as_list()
            reshape = tf.reshape(pool_layer2,
                                 [shape[0], shape[1] * shape[2] * shape[3]])
        with tf.name_scope('Hidden_Layer_1'):
            flat = self.image_size // 4 * self.image_size // 4 * self.num_filters_2
            shape3 = [flat, self.hidden_nodes_1]
            self.fully_hidden_layer_1 = init_weights_bias(shape3, 'Hidden_Layer_1')
            hidden_la1 = tf.nn.relu(linear_activation(reshape,
                                                      self.fully_hidden_layer_1))
        with tf.name_scope('Hidden_Layer_2'):
            shape4 = [self.hidden_nodes_1, self.hidden_nodes_2]
            self.fully_hidden_layer_2 = init_weights_bias(shape4, 'Hidden_Layer_2')
            hidden_la2 = tf.sigmoid(linear_activation(hidden_la1,
                                                      self.fully_hidden_layer_2))
        with tf.name_scope('Hidden_Layer_3'):
            shape5 = [self.hidden_nodes_2, self.hidden_nodes_3]
            self.fully_hidden_layer_3 = init_weights_bias(shape5, 'Hidden_Layer_3')
            hidden_la3 = tf.sigmoid(linear_activation(hidden_la2,
                                                      self.fully_hidden_layer_3))
        with tf.name_scope('Output_Layer'):
            shape6 = [self.hidden_nodes_3, self.num_labels]
            self.fully_hidden_layer_4 = init_weights_bias(shape6, 'Output_Layer')
            self.logits = linear_activation(hidden_la3,
                                            self.fully_hidden_layer_4)

    def create_summaries(self):
        """
        histogram summaries for weights
        """
        tf.summary.histogram('weights1_summ',
                             self.conv_layer_1_wb['weights'])
        tf.summary.histogram('weights2_summ',
                             self.conv_layer_2_wb['weights'])
        tf.summary.histogram('weights3_summ',
                             self.fully_hidden_layer_1['weights'])
        tf.summary.histogram('weights4_summ',
                             self.fully_hidden_layer_2['weights'])
        tf.summary.histogram('weights5_summ',
                             self.fully_hidden_layer_3['weights'])
        tf.summary.histogram('weights6_summ',
                             self.fully_hidden_layer_4['weights'])

    def create_loss(self):
        """
        Create the loss function of the model
        """
        with tf.name_scope("loss"):
            soft = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                           labels=self.train_labels)
            self.loss = tf.reduce_mean(soft)
            tf.summary.scalar(self.loss.op.name, self.loss)

    def create_optimizer(self):
        """
        Create the optimization of the model
        """
        with tf.name_scope("optimizer"):
            self.optimizer = sgd_train(self.loss,
                                       self.learning_rate,
                                       self.steps_for_decay,
                                       self.decay_rate)

    def create_predictions(self):
        self.train_prediction = tf.nn.softmax(self.logits,
                                              name='train_network')
        self.train_pred_cls = tf.argmax(self.train_prediction,
                                        dimension=1)
        self.train_labes_cls = tf.argmax(self.train_labels, 1)

    def create_accuracy(self):
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.train_pred_cls,
                                    self.train_labes_cls)
            self.acc_op = tf.reduce_mean(tf.cast(correct_pred, 'float'))
            tf.summary.scalar(self.acc_op.op.name, self.acc_op)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.create_placeholders()
            self.create_logits()
            self.create_summaries()
            self.create_loss()
            self.create_optimizer()
            self.create_predictions()
            self.create_accuracy()


def train_model(model, dataholder, num_steps=10000, show_step=1000):
    batch_size = model.batch_size
    initial_time = time.time()
    train_dataset = dataholder.train_dataset
    train_labels = dataholder.train_labels

    with tf.Session(graph=model.graph) as session:
        summary_writer = tf.summary.FileWriter(log_path, session.graph)
        all_summaries = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {model.train_input:
                         batch_data,
                         model.train_labels: batch_labels}
            start_time = time.time()
            _, l, predictions, acc, summary = session.run([model.optimizer,
                                                           model.loss,
                                                           model.train_prediction,
                                                           model.acc_op,
                                                           all_summaries],
                                                          feed_dict=feed_dict)
            duration = time.time() - start_time
            summary_writer.add_summary(summary, step)
            summary_writer.flush()

            if (step % show_step == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.2f%%" % (acc * 100))
                print('Duration: %.3f sec' % duration)

    general_duration = time.time() - initial_time
    sec = timedelta(seconds=int(general_duration))
    d_time = datetime(1, 1, 1) + sec
    print(' ')
    print("""The duration of the whole training with % s steps
    is %.2f seconds,""" % (num_steps, general_duration))
    print("""which is equal to:
    %d:%d:%d:%d""" % (d_time.day - 1, d_time.hour, d_time.minute, d_time.second),
          end='')
    print(" (DAYS:HOURS:MIN:SEC)")
    print(' ')
    print(log_path)


if __name__ == "__main__":
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data_4d()
    log_path = get_log()
    c = Config()
    d = DataHolder(train_dataset,
                   train_labels,
                   valid_dataset,
                   valid_labels,
                   test_dataset,
                   test_labels)
    m = CNNModel(c, d)
    train_model(m, d, 100, 10)
