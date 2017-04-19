#all imports
from __future__ import print_function
import subprocess
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.metrics import confusion_matrix
from six.moves import range
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#%matplotlib inline

from util import  get_data_4d

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data_4d()


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


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
                 batch_size=128,
                 patch_size=6,
                 image_size=28,
                 num_labels=10,
                 num_channels=1,
                 num_filters_1=16,
                 num_filters_2=32,
                 hidden_nodes_1=60,
                 hidden_nodes_2=40,
                 hidden_nodes_3=20,
                 learning_rate=0.005):
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
            self.build_graph()

    def create_placeholders(self):           
        shape_X = (self.batch_size, self.image_size,self.image_size, self.num_channels)
        shape_Y =(self.batch_size, self.num_labels)
        name_X = 'X'
        name_Y = 'Y'
        self.tf_train_dataset = tf.placeholder(tf.float32, shape =shape_X ,name = name_X)
        self.tf_train_labels = tf.placeholder(tf.float32, shape=shape_Y, name = name_Y)

    def init_weights_biases(self,shape,weights_name,biases_name):
        layer = {\
            'weights':tf.Variable(tf.truncated_normal(shape, stddev=0.1),name=weights_name),
            'biases':tf.Variable(tf.zeros(shape[-1]),name=biases_name)}
        return layer

    def create_all_weights_bias(self):
            shape1 = [self.patch_size,
                      self.patch_size,
                      self.num_channels,
                      self.num_filters_1]
            self.conv_layer_1_wb = self.init_weights_biases(shape1,
                                                  "weights1",
                                                  "biases1")

            shape2 = [self.patch_size,
                      self.patch_size,
                      self.num_filters_1,
                      self.num_filters_2] 
            self.conv_layer_2_wb = self.init_weights_biases(shape2,
                                                  "weights2",
                                                  "biases2")

            shape3 = [self.image_size // 4 * self.image_size // 4 * self.num_filters_2,
                      self.hidden_nodes_1] 
            self.fully_hidden_layer_1 = self.init_weights_biases(shape3,
                                                       "weights3",
                                                       'biases3')
            shape4 = [self.hidden_nodes_1, self.hidden_nodes_2]
            self.fully_hidden_layer_2 = self.init_weights_biases(shape4,
                                                       "weights4",
                                                       'biases4')

            shape5 = [self.hidden_nodes_2, self.hidden_nodes_3] 
            self.fully_hidden_layer_3= self.init_weights_biases(shape5,
                                                      "weights5",
                                                      'biases5')
            shape6 = [self.hidden_nodes_3, self.num_labels]
            self.fully_hidden_layer_4 = self.init_weights_biases(shape6,
                                                       "weights6",
                                                       'biases6')      
    def create_summaries(self):
        #histogram summaries for weights
        tf.summary.histogram('weights1_summ',self.conv_layer_1_wb['weights'])
        tf.summary.histogram('weights2_summ',self.conv_layer_2_wb['weights'])
        tf.summary.histogram('weights3_summ',self.fully_hidden_layer_1['weights'])
        tf.summary.histogram('weights4_summ',self.fully_hidden_layer_2['weights'])
        tf.summary.histogram('weights5_summ',self.fully_hidden_layer_3['weights'])
        tf.summary.histogram('weights6_summ',self.fully_hidden_layer_4['weights'])

    def apply_conv(self,
                       input_tensor,
                       layer,
                       use_pooling=True):  # Use 2x2 max-pooling.
            """
            Create conv layer
            """

            weights = layer['weights']
            biases = layer['biases']

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

            # Add the biases to the results of the convolution.
            # A bias-value is added to each filter-channel.
            conv_layer += biases
            # Use pooling to down-sample the image resolution?
            if use_pooling:
                # This is 2x2 max-pooling, which means that we
                # consider 2x2 windows and select the largest value
                # in each window. Then we move 2 pixels to the next window.
                conv_layer = tf.nn.max_pool(value=conv_layer,
                                            ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME')
            conv_layer = tf.nn.relu(conv_layer)

            return conv_layer         
        
    def linear_activation(self,
                          input_tensor,
                          layer):
        """
        Method to computing linear activation
        """
        return tf.add(tf.matmul(input_tensor, layer['weights']),
                      layer['biases'])

    def forward_prop(self,
                     input_tensor,
                     V_hidden_layer_1,
                     V_hidden_layer_2,
                     V_hidden_layer_3,
                     V_hidden_layer_4,
                     V_hidden_layer_5,
                     V_hidden_layer_6):
        with tf.name_scope('Convolution_1'):
            conv_layer1 = self.apply_conv(input_tensor,V_hidden_layer_1,use_pooling=True)
        with tf.name_scope('Convolution_2'):
            conv_layer2 = self.apply_conv(conv_layer1,V_hidden_layer_2,use_pooling=True)
        with tf.name_scope('Reshape'):
            shape = conv_layer2.get_shape().as_list() 
            reshape = tf.reshape(conv_layer2, [shape[0], shape[1] * shape[2] * shape[3]])
        with tf.name_scope('Hidden_Layer_1'):
            hidden_la1 = tf.nn.relu(self.linear_activation(reshape,V_hidden_layer_3)) 
        with tf.name_scope('Hidden_Layer_2'):
            hidden_la2 = tf.sigmoid(self.linear_activation(hidden_la1,V_hidden_layer_4))
        with tf.name_scope('Hidden_Layer_3'):
            hidden_la3 = tf.sigmoid(self.linear_activation(hidden_la2,V_hidden_layer_5))     
        with tf.name_scope('Output_Layer'):
            logits = self.linear_activation(hidden_la3,V_hidden_layer_6) 
        return logits

    def create_loss(self):
        """
        Create the loss function of the model
        """
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                               labels=self.tf_train_labels))
            tf.summary.scalar(self.loss.op.name,self.loss)
 
    def sgd_train(self,
                  loss,
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

    def create_optimizer(self):
        """
        Create the optimization of the model
        """
        with tf.name_scope("train"):
            self.optimizer = self.sgd_train(self.loss,
                                            0.9,
                                            100,
                                            0.96)         

    def create_predictions(self):
        self.train_prediction = tf.nn.softmax(self.logits, name='train_network')
        self.train_pred_cls = tf.argmax(self.train_prediction, dimension=1)
        self.valid_prediction = tf.nn.softmax(self.valid_network, name='valid_network')
        self.test_prediction = tf.nn.softmax(self.test_network, name='test_network')

    def create_accuracy(self):
        with tf.name_scope('accuracy'):
            self.train_prediction = tf.nn.softmax(self.logits, name='train_network')
            correct_pred = tf.equal(tf.argmax(self.train_prediction, 1), tf.argmax(self.tf_train_labels, 1))
            self.acc_op = tf.reduce_mean(tf.cast(correct_pred,'float'))        
        
    def build_graph(self):
        self.graph = tf.Graph() 
        with self.graph.as_default():

            #Placeholders
            self.create_placeholders()
            #Constants
            self.tf_valid_dataset = tf.constant(self.valid_dataset, name ='X_va')
            self.tf_test_dataset = tf.constant(self.test_dataset, name ='X_test')
            #all layers weights and biases
            self.create_all_weights_bias()
            self.t1 = self.conv_layer_1_wb['weights']
            self.t2 = self.conv_layer_1_wb['biases']
            self.tt1 = self.conv_layer_2_wb['weights']
            self.tt2 = self.conv_layer_2_wb['biases']
            self.tc1 = self.fully_hidden_layer_1['weights']
            self.tc2 = self.fully_hidden_layer_1['biases']
            self.tcc1 = self.fully_hidden_layer_2['weights']
            self.tcc2 = self.fully_hidden_layer_2['biases']
            self.te1 = self.fully_hidden_layer_3['weights']
            self.te2 = self.fully_hidden_layer_3['biases']
            self.tee1 = self.fully_hidden_layer_4['weights']
            self.tee2 = self.fully_hidden_layer_4['biases']
            #all summaries
            self.create_summaries()
            #forward propagation
            self.logits = self.forward_prop(self.tf_train_dataset,
                                  self.conv_layer_1_wb,
                                  self.conv_layer_2_wb,
                                  self.fully_hidden_layer_1,
                                  self.fully_hidden_layer_2,
                                  self.fully_hidden_layer_3,
                                  self.fully_hidden_layer_4)
            
            self.valid_network = self.forward_prop(self.tf_valid_dataset,
                                       self.conv_layer_1_wb,
                                       self.conv_layer_2_wb,
                                       self.fully_hidden_layer_1,
                                       self.fully_hidden_layer_2,
                                       self.fully_hidden_layer_3,
                                       self.fully_hidden_layer_4)
            
            self.test_network = self.forward_prop(self.tf_test_dataset,
                                       self.conv_layer_1_wb,
                                       self.conv_layer_2_wb,
                                       self.fully_hidden_layer_1,
                                       self.fully_hidden_layer_2,
                                       self.fully_hidden_layer_3,
                                       self.fully_hidden_layer_4)
            #loss
            self.create_loss()
            #opt
            self.create_optimizer()
            #predictions
            self.create_predictions()
            #accuracy
            self.create_accuracy()

#full_test
log_basedir = 'logs'
run_label = time.strftime('%d-%m-%Y_%H-%M-%S') #e.g. 12-11-2016_18-20-45
log_path = os.path.join(log_basedir,run_label)
c = Config()
d = DataHolder(train_dataset,
               train_labels,
               valid_dataset,
               valid_labels,
               test_dataset,
               test_labels)
m = CNNModel(c, d)
num_steps = 10000
batch_size = c.batch_size
initial_time = time.time()

with tf.Session(graph=m.graph) as session:
    summary_writer = tf.summary.FileWriter(log_path, session.graph) 
    all_summaries = tf.summary.merge_all() 
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {m.tf_train_dataset : batch_data, m.tf_train_labels : batch_labels}
        
        #we are going to run the session and count the duration of the running
        start_time = time.time()
        _, l, predictions, acc, summary = session.run(
          [m.optimizer, m.loss, m.train_prediction, m.acc_op, all_summaries], feed_dict=feed_dict)
        duration = time.time() - start_time
       #writing the log
        summary_writer.add_summary(summary,step)
        summary_writer.flush()

        #Printing an overwiew
        if (step % 1000 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.2f%%" % (acc*100))
            print("Validation accuracy: %.1f%%" % accuracy(
            m.valid_prediction.eval(), valid_labels))
            print('Duration: %.3f sec' % duration)
            #print_test_accuracy()

    #after the loop compair our model with the test dataset    
    print("Test accuracy: %.1f%%" % accuracy(m.test_prediction.eval(), test_labels))


general_duration = time.time() - initial_time
sec = timedelta(seconds=int(general_duration))
d_time = datetime(1,1,1) + sec
print(' ')
print('The duration of the whole training with % s steps is %.2f seconds,'\
      % (num_steps,general_duration))
print("which is equal to:  %d:%d:%d:%d" % (d_time.day-1, d_time.hour, d_time.minute, d_time.second), end='')
print(" (DAYS:HOURS:MIN:SEC)")
print(' ')
print(log_path)

