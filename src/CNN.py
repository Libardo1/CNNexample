
import tensorflow as tf
import time
import os
import numpy as np
from datetime import datetime, timedelta
from util import get_data_4d, get_log, randomize_in_place


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
                 patch_size=5,
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
    Apply convolution
    """
    weights = layer['weights']
    bias = layer['bias']
    conv_layer = tf.nn.conv2d(input=input_tensor,
                              filter=weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
    conv_layer += bias
    return conv_layer


def apply_pooling(input_layer):
    """
    Apply max pooling
    """
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
            self.path = get_log()
            self.config = config
            self.test_labels = dataholder.test_labels
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
            shape_input_tensor = (self.batch_size,
                                  self.image_size,
                                  self.image_size,
                                  self.num_channels)
            shape_input_labels = (self.batch_size,
                                  self.num_labels)
            self.input_tensor = tf.placeholder(tf.float32,
                                               shape=shape_input_tensor,
                                               name="input_tensor")
            self.input_labels = tf.placeholder(tf.float32,
                                               shape=shape_input_labels,
                                               name="input_labels")
            shape_one_pic = (1,
                             self.image_size,
                             self.image_size,
                             self.num_channels)
            self.one_pic = tf.placeholder(tf.float32,
                                          shape=shape_one_pic,
                                          name="one_pic")

    def create_constants(self):
        self.TestDataset = tf.constant(self.test_dataset, name='test_data')
        self.TestLabels = tf.constant(self.test_labels, name='test_labels')

    def create_logits(self, input_tensor, Reuse=None):
        with tf.variable_scope('Convolution_1', reuse=Reuse):
            shape1 = [self.patch_size,
                      self.patch_size,
                      self.num_channels,
                      self.num_filters_1]
            self.conv_layer_1_wb = init_weights_bias(shape1, 'Convolution_1')
            conv_layer1 = apply_conv(input_tensor,
                                     self.conv_layer_1_wb)
        with tf.name_scope('Max_pooling1'):
            pool_layer1 = apply_pooling(conv_layer1)
        with tf.variable_scope('Convolution_2', reuse=Reuse):
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
            flat = shape[1] * shape[2] * shape[3]
            reshape = tf.reshape(pool_layer2,
                                 [shape[0], flat])
        with tf.variable_scope('Hidden_Layer_1', reuse=Reuse):
            shape3 = [flat, self.hidden_nodes_1]
            self.hidden_layer_1_wb = init_weights_bias(shape3, 'Hidden_Layer_1')
            linear = linear_activation(reshape, self.hidden_layer_1_wb)
            hidden_layer_1 = tf.nn.relu(linear)
        with tf.variable_scope('Hidden_Layer_2', reuse=Reuse):
            shape4 = [self.hidden_nodes_1, self.hidden_nodes_2]
            self.hidden_layer_2_wd = init_weights_bias(shape4, 'Hidden_Layer_2')
            linear = linear_activation(hidden_layer_1, self.hidden_layer_2_wd)
            hidden_layer_2 = tf.sigmoid(linear)
        with tf.variable_scope('Hidden_Layer_3', reuse=Reuse):
            shape5 = [self.hidden_nodes_2, self.hidden_nodes_3]
            self.hidden_layer_3_wb = init_weights_bias(shape5, 'Hidden_Layer_3')
            linear = linear_activation(hidden_layer_2, self.hidden_layer_3_wb)
            hidden_layer_3 = tf.sigmoid(linear)
        with tf.variable_scope('Output_Layer', reuse=Reuse):
            shape6 = [self.hidden_nodes_3, self.num_labels]
            self.hidden_layer_4_wd = init_weights_bias(shape6, 'Output_Layer')
            logits = linear_activation(hidden_layer_3,
                                       self.hidden_layer_4_wd)
            return logits

    def create_summaries(self):
        """
        histogram summaries for weights
        """
        tf.summary.histogram('weights1_summ',
                             self.conv_layer_1_wb['weights'])
        tf.summary.histogram('weights2_summ',
                             self.conv_layer_2_wb['weights'])
        tf.summary.histogram('weights3_summ',
                             self.hidden_layer_1_wb['weights'])
        tf.summary.histogram('weights4_summ',
                             self.hidden_layer_2_wd['weights'])
        tf.summary.histogram('weights5_summ',
                             self.hidden_layer_3_wb['weights'])
        tf.summary.histogram('weights6_summ',
                             self.hidden_layer_4_wd['weights'])

    def create_loss(self):
        """
        Create the loss function of the model
        """
        with tf.name_scope("loss"):
            soft = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                           labels=self.input_labels)
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
        self.input_prediction = tf.nn.softmax(self.logits,
                                              name='train_network')
        self.input_pred_cls = tf.argmax(self.input_prediction, 1)
        self.train_labes_cls = tf.argmax(self.input_labels, 1)
        test_prediction = tf.nn.softmax(self.test_logits,
                                        name='test_network')
        self.test_prediction = tf.argmax(test_prediction, 1)
        self.test_labes_cls = tf.argmax(self.TestLabels, 1)
        one_pic_prediction = tf.nn.softmax(self.one_pic_logits)
        self.one_pic_prediction_cls = tf.argmax(one_pic_prediction,
                                                1,
                                                name='one_pic_pred')

    def create_accuracy(self):
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.input_pred_cls,
                                    self.train_labes_cls)
            self.acc_op = tf.reduce_mean(tf.cast(correct_pred, 'float'))
            tf.summary.scalar(self.acc_op.op.name, self.acc_op)
            comparison = tf.equal(self.test_prediction, self.test_labes_cls)
            self.acc_test = tf.reduce_mean(tf.cast(comparison, 'float'))

    def create_saver(self):
        self.saver = tf.train.Saver()
        save_dir = 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = os.path.join(save_dir, 'best_validation')

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.create_placeholders()
            self.create_constants()
            self.logits = self.create_logits(self.input_tensor)
            self.test_logits = self.create_logits(self.TestDataset, Reuse=True)
            self.one_pic_logits = self.create_logits(self.one_pic, Reuse=True)
            self.create_summaries()
            self.create_loss()
            self.create_optimizer()
            self.create_predictions()
            self.create_accuracy()
            self.create_saver()


def train_model(model, dataholder, num_steps=10001, show_step=1000):
    log_path = model.path
    batch_size = model.batch_size
    initial_time = time.time()
    train_dataset = dataholder.train_dataset
    train_labels = dataholder.train_labels
    best_acc_test = 0
    marker = ''

    with tf.Session(graph=model.graph) as session:
        summary_writer = tf.summary.FileWriter(log_path, session.graph)
        all_summaries = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        print('Start training')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {model.input_tensor:
                         batch_data,
                         model.input_labels: batch_labels}
            start_time = time.time()
            _, loss, acc, summary = session.run([model.optimizer,
                                                 model.loss,
                                                 model.acc_op,
                                                 all_summaries],
                                                feed_dict=feed_dict)
            duration = time.time() - start_time
            summary_writer.add_summary(summary, step)
            summary_writer.flush()

            if (step % show_step == 0):
                test_acc = session.run(model.acc_test,
                                       feed_dict=feed_dict)
                if test_acc > best_acc_test:
                        best_acc_test = test_acc
                        marker = "*"
                        model.saver.save(sess=session,
                                         save_path=model.save_path)
                print("Minibatch loss at step %d: %f" % (step, loss))
                print("Minibatch accuracy: %.2f%%" % (acc * 100))
                print("Test accuracy: %.2f%%" % (test_acc * 100) + marker)
                print('Duration: %.3f sec' % duration)
                marker = ''

    general_duration = time.time() - initial_time
    sec = timedelta(seconds=int(general_duration))
    d_time = datetime(1, 1, 1) + sec
    print("\n&&&&&&&&& #training steps = {} &&&&&&&&&&&".format(num_steps))
    print("""training time: %d:%d:%d:%d""" % (d_time.day - 1, d_time.hour, d_time.minute, d_time.second), end=' ')
    print("(DAYS:HOURS:MIN:SEC)")
    print("\n&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&")
    print("\ntensorboard  --logdir={}\n".format(log_path))


def check_valid(model, dataholder):
    random_int = np.random.randint(10, size=(1))[0]
    randomize_in_place(dataholder.valid_dataset,
                       dataholder.valid_labels,
                       random_int)
    ValidInputs = dataholder.valid_dataset[: model.batch_size]
    ValidLabels = dataholder.valid_labels[: model.batch_size]
    with tf.Session(graph=model.graph) as session:
                model.saver.restore(sess=session, save_path=model.save_path)
                feed_dict = {model.input_tensor: ValidInputs,
                             model.input_labels: ValidLabels}
                valid_acc = session.run(model.acc_op, feed_dict=feed_dict)
    return valid_acc


def check_test(model):
    with tf.Session(graph=model.graph) as session:
                model.saver.restore(sess=session, save_path=model.save_path)
                test_acc = session.run(model.acc_test)
    return test_acc


def one_prediction(model, input_image):
    with tf.Session(graph=model.graph) as session:
                model.saver.restore(sess=session, save_path=model.save_path)
                feed_dict = {model.one_pic: input_image}
                result = session.run(model.one_pic_prediction_cls,
                                     feed_dict=feed_dict)
    return result[0]


def main():
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data_4d()
    my_config = Config()
    my_dataholder = DataHolder(train_dataset,
                               train_labels,
                               valid_dataset,
                               valid_labels,
                               test_dataset,
                               test_labels)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 1201, 400)
    print("check_valid = ", check_valid(my_model, my_dataholder))
    print("check_test = ", check_valid(my_model, my_dataholder))
    one_example = valid_dataset[0]
    one_example = one_example.reshape(1,
                                      one_example.shape[0],
                                      one_example.shape[1],
                                      one_example.shape[2])
    prediction = chr(one_prediction(my_model, one_example) + ord('A'))
    real = chr(np.argmax(valid_labels[0]) + ord('A'))
    print("Prediction = {}".format(prediction))
    print("Real label = {}".format(real))


if __name__ == "__main__":
    main()
