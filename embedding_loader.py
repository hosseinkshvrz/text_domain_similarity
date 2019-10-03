from gensim.models import Word2Vec
import numpy as np


def load_embeddings(path, data_path, wv_dim):
    model = Word2Vec.load(path + 'word2vec.model')
    word_vectors = model.wv
    print("Number of word vectors: {}".format(len(word_vectors.vocab)))

    MAX_NB_WORDS = len(word_vectors.vocab)
    nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))

    wv_matrix = (np.random.rand(nb_words, wv_dim) - 0.5) / 5.0

    with open(data_path + 'word_index.txt') as file:
        for line in file:
            word, i = line.split(': ')
            i = int(i[:-1])
            if i >= MAX_NB_WORDS:
                continue
            try:
                embedding_vector = word_vectors[word]
                wv_matrix[i] = embedding_vector
            except:
                pass

    return wv_matrix, nb_words
