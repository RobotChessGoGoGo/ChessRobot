import os
import random
from shutil import copyfile
import pandas


"""
Split the training dataset to two parts with 8:2.
"""

# random.seed(2021)

data_path = './data/datasets/train'

test_path = './data/datasets/test'
if not os.path.exists(test_path):
    os.makedirs(test_path)

data_list = os.listdir(data_path)
random.shuffle(data_list)
test_numbers = int(len(data_list) * 0.2)
test_list = data_list[:test_numbers]

for file_name in test_list:
    source_name = os.path.join(data_path, file_name)
    destination_name = os.path.join(test_path, file_name)
    print("Copy from ", source_name, " to ", destination_name)
    copyfile(source_name, destination_name)
    print("Delete ", source_name)
    os.remove(source_name)

print("Done")