import os

import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data/divar/')


def split_data(data_path, max_seq_length):
    sequences1 = []
    sequences2 = []
    scores = []
    lineno = 0
    with open(data_path + 'ad_pairs.txt') as file:
        print('file opened ...')
        for line in file:
            sequence1, sequence2, score = line.split('\t')
            sequence1 = [int(item) for item in sequence1[1:-1].split(', ')]
            if len(sequence1) > max_seq_length:
                sequence1 = sequence1[:max_seq_length]
            elif len(sequence1) < max_seq_length and len(sequence1) != 0:
                for j in range(max_seq_length - len(sequence1)):
                    sequence1.append(sequence1[j % len(sequence1)])
            sequence2 = [int(item) for item in sequence2[1:-1].split(', ')]
            if len(sequence2) > max_seq_length:
                sequence2 = sequence2[:max_seq_length]
            elif len(sequence2) < max_seq_length and len(sequence2) != 0:
                for j in range(max_seq_length - len(sequence2)):
                    sequence2.append(sequence2[j % len(sequence2)])
            score = int(score)

            sequences1.append(sequence1)
            sequences2.append(sequence2)
            scores.append(score)
            lineno += 1
            if lineno % 1000000 == 0:
                print('content of file ', (lineno // 1000000), ' finished ...')

                sequences1 = np.array(sequences1, dtype='int32')
                sequences2 = np.array(sequences2, dtype='int32')
                scores = np.array(scores, dtype='int32')

                print('Shape of data1 train tensor:', sequences1.shape)
                print('Shape of data2 train tensor:', sequences2.shape)
                print('Shape of score train tensor:', scores.shape)

                np.save(data_path + 'sent1_file' + str(lineno // 1000000) + '.npy', sequences1)
                np.save(data_path + 'sent2_file' + str(lineno // 1000000) + '.npy', sequences2)
                np.save(data_path + 'score_file' + str(lineno // 1000000) + '.npy', scores)

                print('files for ', str(lineno // 1000000), 'th iteration saved ...')

                sequences1 = []
                sequences2 = []
                scores = []


MAX_SEQUENCE_LENGTH = 40
split_data(data_path, MAX_SEQUENCE_LENGTH)
