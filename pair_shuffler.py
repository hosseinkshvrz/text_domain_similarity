import os
import random as ra

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data/divar/')

with open(data_path + 'new_ad_pairs.txt') as file:
    content = file.readlines()

length = len(content)
with open(data_path + 'ad_pairs.txt', 'a') as file:
    for i in range(length):
        index = ra.randint(0, len(content) - 1)
        line = content.pop(index)
        file.write(line)
