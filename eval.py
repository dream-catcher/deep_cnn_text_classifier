#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import time
import json
import datetime
import data_helpers
from cnn_model import DeepCNN
from tensorflow.contrib import learn
import csv
import json

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_data_file", "./data/labeled_test_data.csv", "financing classfication test corpus")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("model_dir", "model_dir", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("query_len_limit", 100, "query length limit to unigram")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


MAPPING_FILE_NAME = "label.json" 

OTHER_TYPE = 1

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

x_raw, y_test = data_helpers.load_financing_corpus(FLAGS.model_dir, FLAGS.test_data_file, FLAGS.query_len_limit, train_mode=False)
y_test = data_helpers.convert_one_hot(FLAGS.model_dir, y_test)
y_test = np.argmax(y_test, axis=1)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.model_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))


print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, "checkpoints"))
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
        probability_t = graph.get_operation_by_name("output/probability").outputs[0]


        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for x_test_batch in batches:
            probability, b_predictions = sess.run([probability_t, predictions], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, b_predictions])
            
            if all_probabilities is None:
                all_probabilities = probability
            else:
                all_probabilities = np.concatenate([all_probabilities, probability])
            #print("all_probabilities:{}".format(all_probabilities))
            #print("all_probabilities shape {}:{}".format(len(all_probabilities), len(all_probabilities[0])))


#-------------------------------------------
#Analyze prediction accuracy / precision / recall/ f1_score
y_labels,y_label_to_index = json.load(open(os.path.join(FLAGS.model_dir, MAPPING_FILE_NAME), "r"))
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    # calculate recall rate
    recall_total = 0
    recall_correct = 0
    precision_total = 0
    precision_correct = 0
    recall_cls_map = {}
    precision_cls_map = {}
    for y,p in zip(y_test, all_predictions):
        p = int(p)
        if y != OTHER_TYPE:
            recall_total += 1
            if y not in recall_cls_map:
                recall_cls_map[y] = [1, 0]
            else:
                recall_cls_map[y][0] += 1
            if y == p:
                recall_correct += 1
                recall_cls_map[y][1] += 1
                
        if p != OTHER_TYPE:
            precision_total += 1
            if p not in precision_cls_map:
                precision_cls_map[p] = [1, 0]
            else:
                precision_cls_map[p][0] += 1
            if y == p:
                precision_correct += 1
                precision_cls_map[p][1] += 1
    recall_rate = float(recall_correct) / float(recall_total)
    precision = float(precision_correct) / float(precision_total)
    #f1_score = 2 * 1 / (1 / recall_rate + 1 / precision)
    f1_score = 2 * recall_rate * precision / (recall_rate + precision)
    print("-----------------------")
    print("Total number of recall examples: {}".format(recall_total))
    print("Recall rate:{:g}".format(recall_rate))
    print("Total number of precision examples: {}".format(recall_total))
    print("Precision:{:g}".format(precision))
    print("F1-score:{:g}".format(f1_score))
    print("-----------------------")
    for cls in recall_cls_map:
        recall_total, recall_correct = recall_cls_map[cls]
        recall_rate = float(recall_correct) / float(recall_total)
        print("class:{} recall_rate:{:g}({}/{})".format(cls, recall_rate, recall_correct, recall_total))
    for cls in precision_cls_map:
        precision_total, precision_correct = precision_cls_map[cls]
        precision = float(precision_correct) / float(precision_total)
        #f1_score = 2 * 1 / (1 / recall_rate + 1 / precision)
        f1_score = 2 * recall_rate * precision / (recall_rate + precision)
        print("class:{} precision:{:g}({}/{})".format(cls, precision, precision_correct, precision_total))
        #precision_total, precision_correct = precision_cls_map[cls]
        #precision = float(precision_correct) / float(precision_total)
        #f1_score = 2 * 1 / (1 / recall_rate + 1 / precision)
        #print("class:{:<8}({})\trecall:{:<10g}({:<4}/{:<4})\tprecision:{:<10g}({:<4}/{:<4})\tf1_score:{:<10g}".format(y_labels[cls],
        #    cls, recall_rate, recall_correct, recall_total, precision, precision_correct, precision_total, f1_score))


#Analyze detailed prediction case
predictions_error = []
predictions_list = []
recall_predictions_list = []
recall_predictions_err_list = []
precision_predictions_list = []
precision_predictions_err_list = []
predictions_analysis = np.column_stack((all_predictions == y_test , y_test, all_predictions, all_probabilities))
#print("predictions_analysis:{}".format(predictions_analysis))
for idx, line in enumerate(predictions_analysis):
    real_idx = int(line[1])
    real_label = y_labels[real_idx]
    pred_idx = int(line[2])
    pred_label = y_labels[pred_idx]
    probability_dist = line[3:]
    real_prob = probability_dist[real_idx]
    pred_prob = probability_dist[pred_idx]
    output_tuple = (real_label,real_prob,  pred_label, pred_prob, x_raw[idx])
    predictions_list.append(output_tuple)
    if int(line[0]) == 0:
        predictions_error.append(output_tuple)

    # output recall prediction & error case
    if real_idx != OTHER_TYPE:
        recall_predictions_list.append(output_tuple)
        if int(line[0]) == 0:
            recall_predictions_err_list.append(output_tuple)
        
    # output precision prediction & error case
    if pred_idx != OTHER_TYPE:
        precision_predictions_list.append(output_tuple)
        if int(line[0]) == 0:
            precision_predictions_err_list.append(output_tuple)
# Save the evaluation to a csv
#predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.model_dir, "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_list)

#print("predictions_error:{}".format(predictions_error))
err_path = os.path.join(FLAGS.model_dir, "prediction_err.csv")
print("Saving evaluation to {0}".format(err_path))
with open(err_path, "w") as f:
    csv.writer(f).writerows(predictions_error)

# output recall predictions
out_path = os.path.join(FLAGS.model_dir, "recall_prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(recall_predictions_list)


out_path = os.path.join(FLAGS.model_dir, "recall_prediction_err.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(recall_predictions_err_list)

# output precision
out_path = os.path.join(FLAGS.model_dir, "precision_prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(precision_predictions_list)


out_path = os.path.join(FLAGS.model_dir, "precision_prediction_err.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(precision_predictions_err_list)

