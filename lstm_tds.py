import os

from sklearn.metrics import f1_score, accuracy_score

from data_loader import load_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, multiply, subtract, Lambda, concatenate, Dense, Reshape, LSTM
from keras.backend import abs
from embedding_loader import load_embeddings
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data/divar/')

data1, data2, scores, valid1, valid2, valid_scores, test1, test2, test_scores = load_data()

print(data1.shape)
print(data2.shape)
print(scores.shape)
print(valid1.shape)
print(valid2.shape)
print(valid_scores.shape)
print(test1.shape)
print(test2.shape)
print(test_scores.shape)

WV_DIM = 300
wv_matrix, nb_words = load_embeddings(path, data_path, WV_DIM)
MAX_SEQUENCE_LENGTH = 40

wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=True)

sent1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences1 = wv_layer(sent1_input)

sent2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences2 = wv_layer(sent2_input)

lstmed_sequences1 = LSTM(300)(embedded_sequences1)
lstmed_sequences2 = LSTM(300)(embedded_sequences2)

x_mult_y = multiply([lstmed_sequences1, lstmed_sequences2])
x_minus_y = subtract([lstmed_sequences1, lstmed_sequences2])
abs_x_minus_y = Lambda(lambda x: abs(x))(x_minus_y)

concatenation = concatenate([abs_x_minus_y, x_mult_y])

fcnn_input = Reshape((600,))(concatenation)

fcnn_layer_one = Dense(len(scores[0]), input_shape=(600,), activation='softmax')(fcnn_input)
model = Model(inputs=[sent1_input, sent2_input], outputs=[fcnn_layer_one])

print(model.summary())

filepath = path + 'lstm_weights.last.hdf5'
exists = os.path.isfile(filepath)
if exists:
    model.load_weights(filepath)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             mode='auto')

model.fit([data1, data2],
          scores,
          validation_data=([valid1, valid2], valid_scores),
          epochs=5,
          batch_size=1000,
          callbacks=[checkpoint])

prediction = model.predict([test1, test2])
prediction = np.argmax(prediction, axis=1)
prediction = prediction.tolist()
print(prediction[:10])
test_scores = np.argmax(test_scores, axis=1)
test_scores = test_scores.tolist()
print(test_scores[:10])
acc_score = accuracy_score(test_scores, prediction)
print(acc_score)
f1_score_avg = f1_score(test_scores, prediction, average='weighted')
print(f1_score_avg)
f1_score_list = f1_score(test_scores, prediction, average=None)
print(f1_score_list)
