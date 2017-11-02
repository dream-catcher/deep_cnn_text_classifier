import math
import tensorflow as tf
import numpy as np


class DeepCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by multiple convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size=128, filter_size=3, num_filters=128, cnn_layer_num=3, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        print("inputs:{}".format(self.embedded_chars_expanded))
        # Create a convolution + maxpool layer for first layer
        with tf.name_scope("conv-maxpool-1layer"):
            filter_size = 3
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W,
                strides=[1, 1, embedding_size, 1],
                padding="SAME",
                name="conv")
            print("conv:{}".format(conv))
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 3, 1, 1],
                strides=[1, 3, 1, 1],
                padding="SAME",
                name="pool")
            print("pooled:{}".format(pooled))

        pool_height = math.ceil(sequence_length / 3 )
        cnn_inputs = tf.reshape(pooled, [-1, pool_height, num_filters])
        print("cnn_inputs:{}".format(cnn_inputs))
        with tf.variable_scope("cnn_a"):
            cnn_a_output = cnn_inputs
            for layer_idx in range(cnn_layer_num):
                next_layer = tf.layers.conv1d(
                    inputs=cnn_a_output,
                    filters=num_filters,
                    kernel_size=filter_size,
                    padding="SAME")
                # residual connection
                next_layer += cnn_a_output
                cnn_a_output = tf.tanh(next_layer)
        print("cnn_a_output:{}".format(cnn_a_output))

        # Combine all output neurons
        flat_num = pool_height * num_filters
        self.h_pool_flat = tf.reshape(cnn_a_output, [-1, flat_num])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[flat_num, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.probability = tf.nn.softmax(self.scores, name="probability")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
