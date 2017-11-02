# -*- coding:utf-8 -*-
import numpy as np
import os
import re
import json
import pickle
import itertools
import codecs
from collections import Counter

import conf
if not conf.LOCAL_DEBUG:
    import pydoop
    import pydoop.hdfs


MAPPING_FILE_NAME = "label.json"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


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


def load_financing_corpus(train_dir, data_file, query_len_limit, train_mode=True):
    """
    Load financing corpus classification file
    Format:1)label 2)query   
    """
    print("load_financing_corpus")
    x_text = []
    y = []
    with codecs.open(data_file, encoding="utf-8") as fi:
        for line in fi:
            parts = line.strip().split(",", 2)
            if len(parts) != 2:
                continue
            x_text.append(process_to_unigram(parts[1], query_len_limit))
            y.append(parts[0])

    if train_mode:
        y_labels = sorted(list(set(y)))
        y_label_to_index = {y:i for i,y in enumerate(y_labels)}
        path = os.path.join(train_dir, MAPPING_FILE_NAME)
        #pickle.dump((y_labels,y_label_to_index), open("label_to_index.pkl", "wb"))
        #label_mapping_data = json.dump([y_labels, y_label_to_index])
        #pickle.dump((y_labels,y_label_to_index), pydoop.hdfs.open(path, "w"))
        #pydoop.hdfs.dump(label_mapping_data, pydoop.hdfs.open(path, "w"))
        json.dump([y_labels, y_label_to_index], open(path, "w"))
        #pydoop.hdfs.put(MAPPING_FILE_NAME, path)
        print("dump mapping data")
    return x_text,y

def convert_one_hot(train_dir, y):
    # Process label into one-hot vector
    path = os.path.join(train_dir, MAPPING_FILE_NAME)
    #y_labels, y_label_to_index = pickle.load(pydoop.hdfs.open(path, "r"))
    #y_labels, y_label_to_index = pickle.load(open("label_to_index.pkl", "rb"))
    y_labels, y_label_to_index = json.load(open(path, "r"))
    y_shape = [len(y), len(y_labels)]
    y_one_hot = np.zeros(y_shape)
    print("y_label_to_index:{}".format(y_label_to_index))
    y_index = [y_label_to_index[yl] for yl in y]
    y_one_hot[np.arange(y_shape[0]), y_index] = 1
    return y_one_hot

def convert_one_hot_infer(train_dir, y):
    # Process label into one-hot vector
    path = os.path.join(train_dir, MAPPING_FILE_NAME)
    #y_labels, y_label_to_index = pickle.load(pydoop.hdfs.open(path, "r"))
    #y_labels, y_label_to_index = pickle.load(open("label_to_index.pkl", "rb"))
    y_labels, y_label_to_index = json.load(pydoop.hdfs.open(path, "r"))
    y_shape = [len(y), len(y_labels)]
    y_one_hot = np.zeros(y_shape)
    print("y_label_to_index:{}".format(y_label_to_index))
    y_index = [y_label_to_index[yl] for yl in y]
    y_one_hot[np.arange(y_shape[0]), y_index] = 1
    return y_one_hot

def process_to_unigram(query, query_len_limit):
    """
    process chinese query into unigram split sentences
    english word reserved as one word.
    white space is for delimiter
    """
    if query is None or query == "":
        return ""
    char_list = list(query)
    result = []
    temp = []
    for c in char_list:
        if re.match("[#0-9a-zA-Z_-]", c):
            temp.append(c)
        else:
            if len(temp) > 0:
                result.append(''.join(temp).strip())
                temp = []
            result.append(c.strip())
    start = len(result) - query_len_limit
    start = start if start > 0 else 0
    sent_len = len(result[start:])
    if sent_len > query_len_limit:
        print("{}:{}".format(sent_len, result[start:]))
    ret = ' '.join(result[start:])
    if len(ret.split(' ')) > query_len_limit:
        print("split:{}:{}".format(sent_len, result[start:]))
    return ret
