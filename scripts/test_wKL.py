import os
from collections import defaultdict
import math
import json
# import scipy
import numpy as np
import argparse
import os

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--data_dir', type=str, default="data/ptb")

config, unparsed = parser.parse_known_args()

def tokenize(str):
    if "'s" in str:
        str = str.replace("'s", " 's")
    return str.strip().split()

dir = config.model_dir
data_dir = config.data_dir
training_file = os.path.join(data_dir, "ptb.train.txt")
real_test_file = os.path.join(data_dir, "ptb.test.txt")

def cal_word_freq(file):
    all_word = defaultdict(int)
    tot_token = 0
    for s in file:
        for w in tokenize(s): # .strip().split():
            all_word[w] += 1
            tot_token += 1
    for w in all_word:
        all_word[w] = all_word[w] / tot_token
    return all_word

def cross_entropy(word_dis_1, word_dis_2):
    kl = 0.0
    ce = 0.0

    for w, f in word_dis_2.items():
        if w not in word_dis_1:
            continue

        kl += f * math.log(f / word_dis_1[w])
        ce += f * math.log(word_dis_1[w])
    return kl, ce

training_wf = cal_word_freq(open(training_file))
real_test_wf = cal_word_freq(open(real_test_file))

sampling_files = [f for f in os.listdir(dir) if "-sampling.txt" in f]
sampling_files.sort()
sampling_file = sampling_files[-1]  # [-1]
sampling_file = open(os.path.join(dir, sampling_file))

all_text = sampling_file.readlines()
train, valid, test = all_text[:40000], all_text[40000:43000], all_text[43000:]
test_wf = cal_word_freq(train)
kl, ce = cross_entropy(training_wf, test_wf)
print("kl =", kl)