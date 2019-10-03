import os
import numpy as np
from keras.utils import to_categorical

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data/divar/')

partition = {
    'train1': ['sent1_file2'
        , 'sent1_file3', 'sent1_file4', 'sent1_file6', 'sent1_file7',
               'sent1_file8', 'sent1_file9', 'sent1_file11', 'sent1_file12', 'sent1_file13', 'sent1_file14',
               'sent1_file16', 'sent1_file17', 'sent1_file18', 'sent1_file19', 'sent1_file20'
               ],
    'train2': ['sent2_file2'
        , 'sent2_file3', 'sent2_file4', 'sent2_file6', 'sent2_file7',
               'sent2_file8', 'sent2_file9', 'sent2_file11', 'sent2_file12', 'sent2_file13', 'sent2_file14',
               'sent2_file16', 'sent2_file17', 'sent2_file18', 'sent2_file19', 'sent2_file20'
               ],
    'valid1': ['sent1_file1'
        , 'sent1_file10'
               ],
    'valid2': ['sent2_file1'
        , 'sent2_file10'
               ],
    'test1': ['sent1_file5', 'sent1_file15'],
    'test2': ['sent2_file5', 'sent2_file15']
}

labels = {
    'sent1_file1': 'score_file1', 'sent1_file2': 'score_file2', 'sent1_file3': 'score_file3',
    'sent1_file4': 'score_file4', 'sent1_file5': 'score_file5', 'sent1_file6': 'score_file6',
    'sent1_file7': 'score_file7', 'sent1_file8': 'score_file8', 'sent1_file9': 'score_file9',
    'sent1_file10': 'score_file10', 'sent1_file11': 'score_file11', 'sent1_file12': 'score_file12',
    'sent1_file13': 'score_file13', 'sent1_file14': 'score_file14', 'sent1_file15': 'score_file15',
    'sent1_file16': 'score_file16', 'sent1_file17': 'score_file17', 'sent1_file18': 'score_file18',
    'sent1_file19': 'score_file19', 'sent1_file20': 'score_file20'
}


def load_data():
    data1 = np.load(data_path + partition['train1'][0] + '.npy')
    scores = np.load(data_path + labels[partition['train1'][0]] + '.npy')
    for i, file in enumerate(partition['train1']):
        if i == 0:
            continue
        data1 = np.append(data1, np.load(data_path + file + '.npy'), axis=0)
        scores = np.append(scores, np.load(data_path + labels[file] + '.npy'), axis=0)

    data2 = np.load(data_path + partition['train2'][0] + '.npy')
    for i, file in enumerate(partition['train2']):
        if i == 0:
            continue
        data2 = np.append(data2, np.load(data_path + file + '.npy'), axis=0)

    valid1 = np.load(data_path + partition['valid1'][0] + '.npy')
    valid_scores = np.load(data_path + labels[partition['valid1'][0]] + '.npy')
    for i, file in enumerate(partition['valid1']):
        if i == 0:
            continue
        valid1 = np.append(valid1, np.load(data_path + file + '.npy'), axis=0)
        valid_scores = np.append(valid_scores, np.load(data_path + labels[file] + '.npy'), axis=0)

    valid2 = np.load(data_path + partition['valid2'][0] + '.npy')
    for i, file in enumerate(partition['valid2']):
        if i == 0:
            continue
        valid2 = np.append(valid2, np.load(data_path + file + '.npy'), axis=0)

    test1 = np.load(data_path + partition['test1'][0] + '.npy')
    test_scores = np.load(data_path + labels[partition['test1'][0]] + '.npy')
    for i, file in enumerate(partition['test1']):
        if i == 0:
            continue
        test1 = np.append(test1, np.load(data_path + file + '.npy'), axis=0)
        test_scores = np.append(test_scores, np.load(data_path + labels[file] + '.npy'), axis=0)

    test2 = np.load(data_path + partition['test2'][0] + '.npy')
    for i, file in enumerate(partition['test2']):
        if i == 0:
            continue
        test2 = np.append(test2, np.load(data_path + file + '.npy'), axis=0)

    scores = to_categorical(scores)
    valid_scores = to_categorical(valid_scores)
    test_scores = to_categorical(test_scores)

    return data1, data2, scores, valid1, valid2, valid_scores, test1, test2, test_scores
