import os

import pandas as pd
from collections import Counter
import random as ra

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data/divar/')

with open(data_path + 'token_sentences_list.txt') as file:
    words = file.readlines()

words = [words[i][2:-3].split("', '") for i in range(len(words))]
vocab = Counter()
for text in words:
    vocab.update(text)

with open(data_path + 'word_index.txt') as file:
    content = file.readlines()

word_index = dict()
for i in range(len(content)):
    key, value = content[i].split(': ')
    value = value[:-1]
    if i < 10:
        print(value)
    word_index[key] = int(value)

ads_token_sentences = [sentence for sentence in words[3621689:-200000]]

sequences = []
for i in range(len(ads_token_sentences)):
    if i < 10:
        print(ads_token_sentences[i])
    sequence = [word_index.get(t, 0) for t in ads_token_sentences[i]]
    sequences.append(tuple(sequence))

print('length of sequences: ', len(sequences))

train_df = pd.read_csv(data_path + 'divar_posts_dataset.csv', ',')

cats1 = train_df['cat1'].fillna('').tolist()
cats2 = train_df['cat2'].fillna('').tolist()
cats3 = train_df['cat3'].fillna('').tolist()

cats = []
for i in range(len(cats1)):
    cats.append(cats1[i] + ', ' + cats2[i] + ', ' + cats3[i])
    if i < 10:
        print(cats[i])

print('length of cats: ', len(cats))

ads = dict(zip(sequences, cats))

sequences.sort(key=lambda s: len(s))

min_length = len(sequences[0])
max_length = len(sequences[-1])

print('minimum length, maximum length: ', min_length, ', ', max_length)

start = 0
end = len(sequences)

for i in range(0, len(sequences)):
    if len(sequences[i]) == 4:
        start = i
        break

for i in range(len(sequences)-1, 0, -1):
    if len(sequences[i]) == 50:
        end = i + 1
        break

print('start, end: ', start, ', ', end)
sequences = sequences[start:end]


def score_converter(ad1_cat, ad2_cat):
    ad1_cat = ad1_cat.split(', ')
    ad2_cat = ad2_cat.split(', ')
    if ad1_cat[0] == ad2_cat[0] and ad1_cat[1] == ad2_cat[1] and ad1_cat[2] == ad2_cat[2]:
        return 3
    elif ad1_cat[0] == ad2_cat[0] and ad1_cat[1] == ad2_cat[1]:
        return 2
    elif ad1_cat[0] == ad2_cat[0]:
        return 1
    else:
        return 0


print('collecting ad pairs...')

n_desired_pairs_each_score = 5000000
# ad_pairs_scores = dict()
pairs_set = set()

with open(data_path + 'pair_set.txt') as file:
    content = file.readlines()

for i in range(len(content)):
    first_index, second_index = content[i].split(', ')
    pairs_set.add((int(first_index), int(second_index[:-1])))

with open(data_path + 'new_ad_pairs.txt') as file:
    content = file.readlines()

zeros = [x for x in content if x.split('\t')[2] == '0\n']
ones = [x for x in content if x.split('\t')[2] == '1\n']
twos = [x for x in content if x.split('\t')[2] == '2\n']
threes = [x for x in content if x.split('\t')[2] == '3\n']


counter0 = len(zeros)
counter1 = len(ones)
counter2 = len(twos)
counter3 = len(threes)

while counter3 < n_desired_pairs_each_score or counter2 < n_desired_pairs_each_score \
        or counter1 < n_desired_pairs_each_score or counter0 < n_desired_pairs_each_score:

    first_index = ra.randint(0, len(sequences) - 1)
    second_index = ra.randint(0, len(sequences) - 1)

    if first_index > second_index:
        first_index, second_index = second_index, first_index

    temp_set = set()
    temp_set.add((first_index, second_index))

    if temp_set & pairs_set:
        continue

    pairs_set.add((first_index, second_index))

    if len(sequences[first_index]) - len(sequences[second_index]) > round(0.15 * len(sequences[second_index])) \
            or len(sequences[second_index]) - len(sequences[first_index]) > round(0.15 * len(sequences[first_index])):
        continue

    ad_sentence1 = sequences[first_index]
    ad_sentence2 = sequences[second_index]

    ad1_cat = ads[ad_sentence1]
    ad2_cat = ads[ad_sentence2]

    score = score_converter(ad1_cat, ad2_cat)

    if score == 0 and counter0 < n_desired_pairs_each_score:
        with open(data_path + 'new_ad_pairs.txt', 'a') as file:
            file.write(str(ad_sentence1))
            file.write('\t')
            file.write(str(ad_sentence2))
            file.write('\t')
            file.write(str(score))
            file.write('\n')

        with open(data_path + 'pair_set.txt', 'a') as file:
            file.write(str(first_index) + ', ' + str(second_index) + '\n')

        # ad_pairs_scores[(ad_sentence1, ad_sentence2)] = score
        counter0 += 1
    elif score == 1 and counter1 < n_desired_pairs_each_score:
        with open(data_path + 'new_ad_pairs.txt', 'a') as file:
            file.write(str(ad_sentence1))
            file.write('\t')
            file.write(str(ad_sentence2))
            file.write('\t')
            file.write(str(score))
            file.write('\n')
        # ad_pairs_scores[(ad_sentence1, ad_sentence2)] = score

        with open(data_path + 'pair_set.txt', 'a') as file:
            file.write(str(first_index) + ', ' + str(second_index) + '\n')

        counter1 += 1
    elif score == 2 and counter2 < n_desired_pairs_each_score:
        with open(data_path + 'new_ad_pairs.txt', 'a') as file:
            file.write(str(ad_sentence1))
            file.write('\t')
            file.write(str(ad_sentence2))
            file.write('\t')
            file.write(str(score))
            file.write('\n')
        # ad_pairs_scores[(ad_sentence1, ad_sentence2)] = score

        with open(data_path + 'pair_set.txt', 'a') as file:
            file.write(str(first_index) + ', ' + str(second_index) + '\n')

        counter2 += 1
    elif score == 3 and counter3 < n_desired_pairs_each_score:
        with open(data_path + 'new_ad_pairs.txt', 'a') as file:
            file.write(str(ad_sentence1))
            file.write('\t')
            file.write(str(ad_sentence2))
            file.write('\t')
            file.write(str(score))
            file.write('\n')
        # ad_pairs_scores[(ad_sentence1, ad_sentence2)] = score

        with open(data_path + 'pair_set.txt', 'a') as file:
            file.write(str(first_index) + ', ' + str(second_index) + '\n')

        counter3 += 1

print(counter0)
print(counter1)
print(counter2)
print(counter3)
