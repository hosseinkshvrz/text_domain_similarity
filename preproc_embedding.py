import os
from collections import Counter

import hazm
from gensim.models import Word2Vec
from nltk import WordPunctTokenizer
from tqdm import tqdm

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data/divar/')

with open(data_path + 'bijankhan_Divar_complete_corpus_no_stop_word.txt') as file:
    all_sentences = file.readlines()

print('normalizing ads...')


def pre_process(text):
    text = text.replace('\n', ' ')
    text = text.replace('..', '.')
    text = text.replace('...', '.')
    text = text.replace('....', '.')
    text = text.replace('.....', '.')
    text = text.replace('......', '.')
    text = text.replace('.......', '.')
    text = text.replace('\t', ' ')
    text = text.replace('?', ' ')
    text = text.replace('؟', ' ')
    text = text.replace('<', ' ')
    text = text.replace('>', ' ')
    text = text.replace('!', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace('_', ' ')
    text = text.replace(':', ' ')
    text = text.replace('/', ' ')
    text = text.replace('|', ' ')
    text = text.replace('|', ' ')
    text = text.replace('»', ' ')
    text = text.replace('«', ' ')
    text = text.replace('؛', ' ')
    text = text.replace(',', ' ')
    text = text.replace('،', ' ')
    text = text.replace('"', ' ')
    text = text.replace('\'', ' ')
    text = text.replace(':', ' ')
    text = text.replace('\\', '')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('$NUM', ' ')
    text = text.replace('•', ' ')
    text = text.replace('*', ' ')
    text = text.replace('✴', ' ')
    text = text.replace('✔', ' ')
    text = text.replace('-', ' ')
    text = text.replace('=', ' ')
    text = text.replace('\u200c', '')
    text = text.replace(' ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('   ', ' ')
    text = text.replace('    ', ' ')
    text = text.replace('     ', ' ')
    text = text.replace('      ', ' ')
    text = text.replace('       ', ' ')
    text = text.replace('        ', ' ')

    return text


normalizer = hazm.Normalizer()

all_sentences = [normalizer.normalize(elem) for elem in all_sentences]
all_sentences = [pre_process(elem) for elem in all_sentences]


tokenizer = WordPunctTokenizer()
vocab = Counter()


def text_to_wordlist(text, lower=False):
    text = tokenizer.tokenize(text)
    if lower:
        text = [t.lower() for t in text]
    vocab.update(text)
    return text


def process_sentences(list_sentences, lower=False):
    words = []
    for text in tqdm(list_sentences):
        txt = text_to_wordlist(text, lower=lower)
        words.append(txt)
    return words


print('tokenizing sentences...')
words = process_sentences(all_sentences, lower=True)
print("The vocabulary contains {} unique tokens".format(len(vocab)))

with open(data_path + 'persian_stop_words.txt') as file:
    content = file.readlines()

stopwords = [line.split('\n')[0] for line in content][1:]
stopwords = [normalizer.normalize(word) for word in stopwords]
stopwords = set(stopwords)

for i in range(len(words)):
    sntnc_set = set(words[i])
    intersections = list(set.intersection(stopwords, sntnc_set))
    words[i] = [word for word in words[i] if word not in intersections]

# save tokenized sentences
with open(data_path + 'token_sentences_list.txt', 'a') as file:
    for i in range(len(words)):
        file.write(str(words[i]))
        file.write('\n')

print('create word embedding vectors...')
model = Word2Vec(words, size=300, window=5, min_count=2, workers=8, sg=0, negative=5)
model.save("word2vec.model")

word_vectors = model.wv
print("Number of word vectors: {}".format(len(word_vectors.vocab)))

MAX_NB_WORDS = len(word_vectors.vocab)

word_index = {t[0]: i + 1 for i, t in enumerate(vocab.most_common(MAX_NB_WORDS))}

with open(data_path + 'word_index.txt', 'a') as file:
    for k in word_index:
        file.write(str(k) + ': ' + str(word_index[k]) + '\n')
