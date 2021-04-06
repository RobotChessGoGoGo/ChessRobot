import pandas as pd
import os

path = './data/datasets/train'
files = os.listdir('./data/datasets/train')

for file in files:
    data = pd.read_csv(os.path.join(path, file), index_col=0)
    if len(data) == 0:
        os.remove(os.path.join(path, file))
        print(file, " is empty! Delete this file!")