
import tensorflow as tf
import time
import os
import numpy as np
from datetime import datetime, timedelta
from util import get_data_4d, get_log_path, timeit
from DataHolder import DataHolder
from Config import Config
from tf_functions import apply_conv, apply_pooling, linear_activation, gd_train, init_wb


class CNNModel:
    """
    CNN model. Architecture: conv-pull-conv-pull-fc
    """

    def __init__(self, config, dataholder):
        """
        :type config: Config
        :type dataholder: DataHolder
        """
        self.path = get_log_path()
        self.config = config
        self.test_labels = dataholder.test_labels
        self.valid_dataset = dataholder.valid_dataset
        self.valid_labels = dataholder.valid_labels
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
        self.dropout = self.config.dropout
        self.mean = self.config.mean
        self.std = self.config.std
        if self.config.tunning:
            self.random_seed = 0
        else:
            self.random_seed = None
        self.build_graph()

    def create_placeholders(self):
        """
        Method to create the placeholders for the graph.
        self.input_tensor and self.input_labels are the
        placeholder for the training; self.one_pic is the placefolder
        for predict just one image
        """
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
        """
        Method to create the constants for the graph.
        """
        self.TestDataset = tf.constant(self.test_dataset, name='test_data')
        self.TestLabels = tf.constant(self.test_labels, name='test_labels')
        self.ValidDataset = tf.constant(self.valid_dataset, name='valid_data')
        self.ValidLabels = tf.constant(self.valid_labels, name='valid_labels')

    def create_logits(self, input_tensor, Reuse=None):
        """
        Method that calculates the forward propagation.
        Reuse is the param to reuse the same weights.

        :type input_tensor: tf tensor
        :type Reuse: tf param >> None or True
        :rtype: tf tensor
        """
        with tf.variable_scope('Convolution_1', reuse=Reuse):
            shape1 = [self.patch_size,
                      self.patch_size,
                      self.num_channels,
                      self.num_filters_1]
            self.conv_layer_1_wb = init_wb(shape1,
                                           'Convolution_1',
                                           self.mean,
                                           self.std,
                                           self.random_seed)
            conv_layer1 = apply_conv(input_tensor,
                                     self.conv_layer_1_wb)
        with tf.name_scope('Max_pooling1'):
            if Reuse is None:
                conv_layer1 = tf.nn.dropout(conv_layer1, self.dropout)
            pool_layer1 = apply_pooling(conv_layer1)
        with tf.variable_scope('Convolution_2', reuse=Reuse):
            shape2 = [self.patch_size,
                      self.patch_size,
                      self.num_filters_1,
                      self.num_filters_2]
            self.conv_layer_2_wb = init_wb(shape2,
                                           'Convolution_2',
                                           self.mean,
                                           self.std,
                                           self.random_seed)
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
            self.hidden_layer_1_wb = init_wb(shape3,
                                             'Hidden_Layer_1',
                                             self.mean,
                                             self.std,
                                             self.random_seed)
            linear = linear_activation(reshape, self.hidden_layer_1_wb)
            hidden_layer_1 = tf.nn.relu(linear)
        with tf.variable_scope('Hidden_Layer_2', reuse=Reuse):
            shape4 = [self.hidden_nodes_1, self.hidden_nodes_2]
            self.hidden_layer_2_wb = init_wb(shape4,
                                             'Hidden_Layer_2',
                                             self.mean,
                                             self.std,
                                             self.random_seed)
            linear = linear_activation(hidden_layer_1, self.hidden_layer_2_wb)
            hidden_layer_2 = tf.sigmoid(linear)
        with tf.variable_scope('Hidden_Layer_3', reuse=Reuse):
            shape5 = [self.hidden_nodes_2, self.hidden_nodes_3]
            self.hidden_layer_3_wb = init_wb(shape5,
                                             'Hidden_Layer_3',
                                             self.mean,
                                             self.std,
                                             self.random_seed)
            linear = linear_activation(hidden_layer_2, self.hidden_layer_3_wb)
            hidden_layer_3 = tf.sigmoid(linear)
        with tf.variable_scope('Output_Layer', reuse=Reuse):
            shape6 = [self.hidden_nodes_3, self.num_labels]
            self.hidden_layer_4_wb = init_wb(shape6,
                                             'Output_Layer',
                                             self.mean,
                                             self.std,
                                             self.random_seed)
            logits = linear_activation(hidden_layer_3,
                                       self.hidden_layer_4_wb)
            return logits

    def create_summaries(self):
        """
        Method to create the histogram summaries for all weights
        """
        tf.summary.histogram('weights1_summ',
                             self.conv_layer_1_wb['weights'])
        tf.summary.histogram('weights2_summ',
                             self.conv_layer_2_wb['weights'])
        tf.summary.histogram('weights3_summ',
                             self.hidden_layer_1_wb['weights'])
        tf.summary.histogram('weights4_summ',
                             self.hidden_layer_2_wb['weights'])
        tf.summary.histogram('weights5_summ',
                             self.hidden_layer_3_wb['weights'])
        tf.summary.histogram('weights6_summ',
                             self.hidden_layer_4_wb['weights'])

    def create_loss(self):
        """
        Method to create the loss function of the graph
        """
        with tf.name_scope("loss"):
            soft = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                           labels=self.input_labels)
            self.loss = tf.reduce_mean(soft)
            tf.summary.scalar(self.loss.op.name, self.loss)

    def create_optimizer(self):
        """
        Method to create the optimizer of the graph
        """
        with tf.name_scope("optimizer"):
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.optimizer = opt.minimize(self.loss)

    def create_optimizer_decay(self):
        """
        Method to create the optimizer of the graph using the training steps
        to change the learning rate
        """
        with tf.name_scope("optimizer"):
            self.optimizer = gd_train(self.loss,
                                      self.learning_rate,
                                      self.steps_for_decay,
                                      self.decay_rate)

    def create_predictions(self):
        """
        Method to create the prediction for the placeholders,
        for the test dataset, for the valid dataset and for the
        single image.
        """
        self.input_prediction = tf.nn.softmax(self.logits,
                                              name='train_logits')
        self.input_pred_cls = tf.argmax(self.input_prediction, 1)
        self.train_labes_cls = tf.argmax(self.input_labels, 1)
        test_prediction = tf.nn.softmax(self.test_logits,
                                        name='test_logits')
        self.test_prediction = tf.argmax(test_prediction, 1)
        self.test_labes_cls = tf.argmax(self.TestLabels, 1)
        valid_prediction = tf.nn.softmax(self.valid_logits,
                                         name='valid_logits')
        self.valid_prediction = tf.argmax(valid_prediction, 1)
        self.valid_labes_cls = tf.argmax(self.ValidLabels, 1)
        one_pic_prediction = tf.nn.softmax(self.one_pic_logits)
        self.one_pic_prediction_cls = tf.argmax(one_pic_prediction,
                                                1,
                                                name='one_pic_pred')

    def create_accuracy(self):
        """
        Method to create the accuracy score of the test dataset
        and the valid.
        """
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.input_pred_cls,
                                    self.train_labes_cls)
            self.acc_op = tf.reduce_mean(tf.cast(correct_pred, 'float'))
            tf.summary.scalar(self.acc_op.op.name, self.acc_op)
            test_comparison = tf.equal(self.test_prediction,
                                       self.test_labes_cls)
            self.acc_test = tf.reduce_mean(tf.cast(test_comparison, 'float'))
            valid_comparison = tf.equal(self.valid_prediction,
                                        self.valid_labes_cls)
            self.acc_valid = tf.reduce_mean(tf.cast(valid_comparison, 'float'))

    def create_saver(self):
        """
        Method to create the graph saver.
        """
        self.saver = tf.train.Saver()
        save_dir = 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = os.path.join(save_dir, 'best_validation')

    def build_graph(self):
        """
        Method to build the computation graph.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.create_placeholders()
            self.create_constants()
            self.logits = self.create_logits(self.input_tensor)
            self.test_logits = self.create_logits(self.TestDataset,
                                                  Reuse=True)
            self.valid_logits = self.create_logits(self.ValidDataset,
                                                   Reuse=True)
            self.one_pic_logits = self.create_logits(self.one_pic,
                                                     Reuse=True)
            self.create_summaries()
            self.create_loss()
            self.create_optimizer()
            self.create_predictions()
            self.create_accuracy()
            self.create_saver()


@timeit([2])
def train_model(model,
                dataholder,
                num_steps=10001,
                show_step=1000,
                verbose=True):
    """
    Function to train the model in num_steps steps. For each step that
    is divisible by show_step this function calculates the accuracy test and
    saves the graph weights if the accuracy test is better than
    the previous one.

    :type model: CNNModel
    :type dataholder: DataHolder
    :type num_steps: int
    :type show_steps: int
    """
    log_path = model.path
    batch_size = model.batch_size
    initial_time = time.time()
    train_dataset = dataholder.train_dataset
    train_labels = dataholder.train_labels
    best_valid_test = 0
    marker = ''

    with tf.Session(graph=model.graph) as session:
        summary_writer = tf.summary.FileWriter(log_path, session.graph)
        all_summaries = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        print('Start training')
        print("{}  {}  {}  {}".format("step",
                                      "batch_acc",
                                      "valid_acc",
                                      "elapsed_time"))
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
                valid_acc = session.run(model.acc_valid)
                if valid_acc > best_valid_test:
                        best_valid_test = valid_acc
                        marker = "*"
                        model.saver.save(sess=session,
                                         save_path=model.save_path)
                print("{:3d}   {:.2f}%        {:.2f}%{:s}    {:.2f}(s)".format(step,
                                                                               acc * 100,
                                                                               valid_acc * 100,
                                                                               marker,
                                                                               duration))
                marker = ''
    if verbose:
        general_duration = time.time() - initial_time
        sec = timedelta(seconds=int(general_duration))
        d_time = datetime(1, 1, 1) + sec
        print("\n&&&&&&&&& #training steps = {} &&&&&&&&&&&".format(num_steps))
        print("""training time: %d:%d:%d:%d""" % (d_time.day - 1, d_time.hour, d_time.minute, d_time.second), end=' ')
        print("(DAYS:HOURS:MIN:SEC)")
        print("\n&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&")
        print("\ntensorboard  --logdir={}\n".format(log_path))


def check_test(model):
    """
    Function that returns the accuracy of the test dataset.

    :type model: CNNModel
    :rtype : float
    """
    with tf.Session(graph=model.graph) as session:
                model.saver.restore(sess=session, save_path=model.save_path)
                acc_test = session.run(model.acc_test)
    return acc_test


def check_valid(model):
    """
    Function that returns the accuracy of the valid dataset.

    :type model: CNNModel
    :rtype : float
    """
    with tf.Session(graph=model.graph) as session:
                model.saver.restore(sess=session, save_path=model.save_path)
                acc_valid = session.run(model.acc_valid)
    return acc_valid


def one_prediction(model, input_image):
    """
    Function that returns the predicted class of the input_image.

    :type model: CNNModel
    :rtype : int
    """
    with tf.Session(graph=model.graph) as session:
                model.saver.restore(sess=session, save_path=model.save_path)
                feed_dict = {model.one_pic: input_image}
                result = session.run(model.one_pic_prediction_cls,
                                     feed_dict=feed_dict)
    return result[0]


def main():
    """
    Basic script that shows the training of the model,
    the accuracy of the test and valid datasets, and
    the prediction of one specific image.
    """
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data_4d()
    # my_config = Config()
    my_config = Config(learning_rate=0.52660658241,
                       dropout=0.75,
                       batch_size=230,
                       steps_for_decay=800,
                       num_filters_1=12,
                       hidden_nodes_3=300,
                       hidden_nodes_1=900,
                       decay_rate=0.349144998004,
                       num_filters_2=24,
                       patch_size=11,
                       image_size=28,
                       hidden_nodes_2=600)
    my_dataholder = DataHolder(train_dataset,
                               train_labels,
                               valid_dataset,
                               valid_labels,
                               test_dataset,
                               test_labels)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 4 * 10001, 1000)
    print("check_valid = ", check_valid(my_model))
    print("check_test = ", check_test(my_model))
    one_example = test_dataset[0]
    one_example = one_example.reshape(1,
                                      one_example.shape[0],
                                      one_example.shape[1],
                                      one_example.shape[2])
    prediction = chr(one_prediction(my_model, one_example) + ord('A'))
    real = chr(np.argmax(test_labels[0]) + ord('A'))
    print("Prediction = {}".format(prediction))
    print("Real label = {}".format(real))


if __name__ == "__main__":
    main()
