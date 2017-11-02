#!/usr/bin/env python
import numpy as np
import sys


def shuffle(fin_name, fout_name):
    print("start!")
    lines = open(fin_name).readlines()
    print("shuffle indices")
    shuffle_indices = np.random.permutation(len(lines))
    fout = open(fout_name, "w")
    print("output lines:")
    for i in shuffle_indices:
        fout.write(lines[i])
    print("finish!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("format:./shuffle.py xxx.csv")
        exit()
    FILE_NAME = sys.argv[1]
    path_fields = FILE_NAME.rsplit("/")
    SHUFFLE_FILE_NAME = path_fields[0] + "/shuffle_" + path_fields[-1]
    print("file_name:{} --> shuffle_name:{}".format(FILE_NAME, SHUFFLE_FILE_NAME))
    shuffle(FILE_NAME, SHUFFLE_FILE_NAME)


