import os
import shutil
from collections import Counter
import math
from tqdm import tqdm

def handle_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), 'r') as f:
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict((idx, label) for idx, label in tokens)
    min_num_train_sample = (Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    num_valid_sample = math.floor(min_num_train_sample * valid_ratio)
    label_count = dict()
    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))
    for train_file in tqdm(os.listdir(os.path.join(data_dir, train_dir))):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        if label not in label_count or label_count[label] < num_valid_sample:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in tqdm(os.listdir(os.path.join(data_dir, test_dir))):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))

data_dir = '../data/ImageNet Dogs'
label_file = 'labels.csv'
train_dir = 'train'
test_dir = 'test'
input_dir = 'train_valid_test'
valid_ratio = 0.1

handle_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)