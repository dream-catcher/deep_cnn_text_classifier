import tensorflow as tf
import numpy as np
import math


class CNNModel():
    """
    CNN model for financial recommendation classification.
    An embedding layer,two convolutional layer, one fc layer 
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="embedding")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print("embedded_chars_expanded:{}".format(self.embedded_chars_expanded))

        # Create first convolution + maxpool layer 
        with tf.name_scope("conv-layer-1"):
            # Convolution Layer
            filter_size1 = 3
            filter_shape1 = [filter_size1, embedding_size, 1, num_filters]
            print("filter_shape1:{}".format(filter_shape1))
            W1 = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="W")
            b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            print("W1:{} b1:{}".format(W1, b1))
            conv1 = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W1,
                strides=[1, 1, embedding_size, 1],
                padding="SAME",
                name="conv1")
            print("conv1 shape:{}".format(conv1))
            # Apply nonlinearity
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
            #h1 = tf.nn.dropout(h1, self.dropout_keep_prob)
            print("h1:shape:{}".format(h1))
            # Maxpooling over the outputs
            pooled1 = tf.nn.max_pool(
                h1,
                ksize=[1, 2, 1, 1],
                strides=[1, 2, 1, 1],
                padding="SAME",
                name="pool")
            print("pooled1:shape:{}".format(pooled1))
                
        # Create second convolution + maxpool layer 
        with tf.name_scope("conv-layer-2"):
            # Convolution Layer
            filter_size2 = 3
            filter_shape2 = [filter_size2, 1, num_filters, num_filters * 2]
            W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W")
            b2 = tf.Variable(tf.constant(0.1, shape=[num_filters * 2]), name="b")
            conv2 = tf.nn.conv2d(
                pooled1,
                W2,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv")
            print("conv2 shape:{}".format(conv2))
            # Apply nonlinearity
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
            #h2 = tf.nn.dropout(h2, self.dropout_keep_prob)
            print("h2:shape:{}".format(h2))
            # Maxpooling over the outputs
            pooled2 = tf.nn.max_pool(
                h2,
                ksize=[1, 2, 1, 1],
                strides=[1, 2, 1, 1],
                padding="SAME",
                name="pool")
            print("pooled2:shape:{}".format(pooled2))

        ## Create third convolution + maxpool layer 
        #with tf.name_scope("conv-layer-3"):
        #    # Convolution Layer
        #    filter_size3 = 3
        #    filter_shape3 = [filter_size3, 1, num_filters * 2, num_filters * 4]
        #    W3 = tf.Variable(tf.truncated_normal(filter_shape3, stddev=0.1), name="W")
        #    b3 = tf.Variable(tf.constant(0.1, shape=[num_filters * 4]), name="b")
        #    conv3 = tf.nn.conv2d(
        #        pooled2,
        #        W3,
        #        strides=[1, 1, 1, 1],
        #        padding="SAME",
        #        name="conv")
        #    print("conv3 shape:{}".format(conv3))
        #    # Apply nonlinearity
        #    h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
        #    print("h3:shape:{}".format(h3))
        #    # Maxpooling over the outputs
        #    pooled3 = tf.nn.max_pool(
        #        h3,
        #        ksize=[1, 2, 1, 1],
        #        strides=[1, 2, 1, 1],
        #        padding="SAME",
        #        name="pool")
        #    print("pooled3:shape:{}".format(pooled3))

        # Fully connected layer
        fc_number = num_filters * 2 * int((((sequence_length + 1) / 2 + 1) / 2))
        #fc_number = num_filters * math.ceil(sequence_length / 8)
        #pool_shape = tf.shape(pooled3)
        #fc_number = pool_shape[1] * pool_shape[2] * pool_shape[3] 
        #fc_number = num_filters * math.ceil(sequence_length / 2)
        #fc_number = num_filters * sequence_length
        self.h_pool_flat = tf.reshape(pooled2, [-1, fc_number])
        print("h_pool_flat:{}".format(self.h_pool_flat))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        print("h_drop:{}".format(self.h_drop))

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[fc_number, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            print("w:{} b:{} -->scores:{}".format(W, b, self.scores))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
