import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd

sampled_train = pd.read_pickle("kaggle_data/notsampled_train.pkl")
tweets_test = pd.read_pickle("kaggle_data/tweets_test.pkl")
sampled_train = sampled_train.sample(frac=1).reset_index(drop=True)

from tensorflow.keras.preprocessing.text import Tokenizer
tok = Tokenizer()
tok.fit_on_texts(pd.concat([sampled_train,tweets_test],ignore_index=True)['text'])
vocab_size = len(tok.word_index) + 1

train_encoded_phrase = tok.texts_to_sequences(sampled_train['text'])
test_encoded_phrase = tok.texts_to_sequences(tweets_test['text'])

from tensorflow.keras.preprocessing.sequence import pad_sequences
X_train= pad_sequences(train_encoded_phrase, maxlen=61, padding='post')
X_test= pad_sequences(test_encoded_phrase, maxlen=61, padding='post')
print(X_train[:5])

y_train = sampled_train['emotion']
y_test = tweets_test['emotion']

from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

def label_encode(le, labels):
    enc = le.transform(labels)
    return np_utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

y_train = label_encode(label_encoder, y_train)
# y_test = label_encode(label_encoder, y_test)

print('\n\n## After convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.20,random_state=42)

import gensim
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews/GoogleNews-vectors-negative300.bin.gz', binary=True)

import numpy as np
words = list(tok.word_index.keys())
missing = 0
embedding_vector_size=300
embedding_matrix = np.zeros((vocab_size, embedding_vector_size))
for index, word in enumerate(words):
    try:
        embedding_matrix[index] = w2v_model[word]
    except:
        missing += 1

print("not in w2v model: {}/{}".format(missing, len(words)))

from tensorflow import keras
from tensorflow.keras import layers
from keras.initializers import Constant

inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(input_dim=vocab_size, output_dim=300, embeddings_initializer=Constant(embedding_matrix))(inputs)
x = layers.SpatialDropout1D(0.8)(x)
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(256))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(8, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=512,epochs=3)

pred_result = model.predict(X_test, batch_size=1000)
print(pred_result[:5])

pred_result = label_decode(label_encoder, pred_result)
print(pred_result[:5])

tweets_test['emotion'] = pred_result
tweets_test['id'] = tweets_test['tweet_id']
results = tweets_test[['id', 'emotion']]
results.to_csv("kaggle_data/6th.csv", index=False)



