#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import json
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pickle

import pydoop

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_data_file", "./data/labeled_test_data.csv", "financing classfication test corpus")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("model-path", "model_dir", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval", True, "Evaluate on all test data")
tf.flags.DEFINE_integer("query_len_limit", 100, "query length limit to unigram")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


MAPPING_FILE_NAME = "label_to_index.json" 

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval:
    x_raw, y_test = data_helpers.load_financing_corpus(FLAGS.model_path, FLAGS.test_data_file, FLAGS.query_len_limit, train_mode=False)
    y_test = data_helpers.convert_one_hot_infer(FLAGS.model_path, y_test)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.model_path, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))


print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.model_path, "checkpoints"))
print("checkpoint_file:" + checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))


#y_labels,y_label_to_index = pickle.load(open(os.path.join(FLAGS.model_path, MAPPING_FILE_NAME), "rb"))
#y_labels,y_label_to_index = json.load(pydoop.hdfs.open(os.path.join(FLAGS.model_path, MAPPING_FILE_NAME), "r"))
#
#predictions_error = []
#predictions_analysis = np.column_stack((all_predictions == y_test , y_test, all_predictions))
#for idx, line in enumerate(predictions_analysis):
#    if int(line[0]) == 0:
#        real_label = y_labels[int(line[1])]
#        pred_label = y_labels[int(line[2])]
#        predictions_error.append((real_label, pred_label, x_raw[idx]))
##print("predictions_error:{}".format(predictions_error))
#err_path = os.path.join(FLAGS.model_path, "prediction_err.csv")
#with pydoop.hdfs.open(err_path, "w") as f:
#    csv.writer(f).writerows(predictions_error)
#        
## Save the evaluation to a csv
#predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
#out_path = os.path.join(FLAGS.model_path, "prediction.csv")
#print("Saving evaluation to {0}".format(out_path))
#with pydoop.hdfs.open(out_path, 'w') as f:
#    csv.writer(f).writerows(predictions_human_readable)
