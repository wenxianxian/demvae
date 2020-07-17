import os
from collections import defaultdict
import math
import json
import shutil
import argparse
import os

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--data_dir', type=str, default="data/ptb")

config, unparsed = parser.parse_known_args()

dir = config.model_dir
data_dir = config.data_dir

train_file_real = os.path.join(data_dir, "ptb.train.txt")
valid_file_real = os.path.join(data_dir, "ptb.valid.txt")
test_file_real = os.path.join(data_dir, "ptb.test.txt")

# Split sampling files
sampling_files = [f for f in os.listdir(dir) if "-sampling.txt" in f]
sampling_files.sort()

if not os.path.isdir(os.path.join(dir, "forward_PPL")):
    os.mkdir(os.path.join(dir, "forward_PPL"))

if not os.path.isdir(os.path.join(dir, "reverse_PPL")):
    os.mkdir(os.path.join(dir, "reverse_PPL"))

sampling_file = open(os.path.join(dir, sampling_files[-1]))

all_ = sampling_file.readlines()
train, valid, test = all_[:40000], all_[40000:43000], all_[43000:]

train_file = open(os.path.join(dir, "reverse_PPL", "train.txt"), "w")
valid_file = open(os.path.join(dir, "reverse_PPL",  "valid.txt"), "w")
test_file = open(os.path.join(dir,"forward_PPL", "ptb.test.txt"), "w")

shutil.copyfile(train_file_real, os.path.join(dir, "forward_PPL", "ptb.train.txt"))
shutil.copyfile(valid_file_real, os.path.join(dir, "forward_PPL", "ptb.valid.txt"))
shutil.copyfile(test_file_real, os.path.join(dir, "reverse_PPL", "test.txt"))

train_file.write("".join(train))
valid_file.write("".join(valid))
test_file.write("".join(test))