from __future__ import print_function
from keras.models import Sequential,Model
from keras.layers import  Dense,Activation,Dropout,LSTM,Input,Flatten,Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.metrics import categorical_accuracy
import numpy as np
import random
import sys
import os
import time
import codecs
import  collections
from six.moves import cPickle
import  string
import  re
string_punctuation = string.punctuation
print(string_punctuation)
string_punctuation = (list(string_punctuation))

import glob

import spacy
nlp = spacy.load('en')
print(nlp)

data_dir = 'data/'
save_dir = 'save/'
seq_length = 30
sequences_step = 1


file_list = ["101","102","103","104","105","106","107","108","109","110","111"]
vocab_file = os.path.join(save_dir,"words_vocab.pkl")

def create_wordlist(doc):
    wl = []
    for word in doc:
        word = word.text.strip()
        if word not in("\n","\n\n","\u2009","\xa0","\n\n\n"):
            if word not in string_punctuation:
                if len(word) != 0:
                    wl.append(word.lower())
    return wl

wordlist = []
for file_name in file_list:
    input_file = os.path.join(data_dir,file_name+".txt")
    # print(input_file)
    with codecs.open(input_file,"r") as f:
        data = f.read()
        data = re.sub(r'[^\w\s]','',data)
    # print(data)
    doc = nlp(data)
    wl = create_wordlist(doc)
    wordlist = wordlist + wl
    # print(wordlist)

    word_counts = collections.Counter(wordlist)
    # print(word_counts)
    vocabulary_inv = [x[0] for x in word_counts.most_common() if len(x[0]) != 0]
    vocabulary_inv = list(sorted((vocabulary_inv)))
    print(vocabulary_inv)
    print(len(vocabulary_inv),"before sorting")

    vocab = {x:i for i,x in enumerate(vocabulary_inv)}
    words = [x[0] for x in word_counts.most_common() if len(x[0]) != 0]

    vocab_size = len(words)
    print(vocab_size,"after sorting")

    with open(os.path.join(vocab_file), 'wb') as f:
        cPickle.dump((words, vocab, vocabulary_inv), f)

    sequences = []
    next_words = []
    for i in range(0, len(wordlist) - seq_length, sequences_step):
        sequences.append(wordlist[i: i + seq_length])
        next_words.append(wordlist[i + seq_length])

    print('nb sequences:', len(sequences))

    X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
    y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, word in enumerate(sentence):
            X[i, t, vocab[word]] = 1
        y[i, vocab[next_words[i]]] = 1


    def bidirectional_lstm_model(seq_length, vocab_size):
        print('Build LSTM model.')
        model = Sequential()
        model.add(Bidirectional(LSTM(rnn_size, activation="relu"), input_shape=(seq_length, vocab_size)))
        model.add(Dropout(0.6))
        model.add(Dense(vocab_size))
        model.add(Activation('softmax'))

        optimizer = Adam(lr=learning_rate)
        callbacks = [EarlyStopping(patience=2, monitor='val_loss')]
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
        return model


    rnn_size = 256  # size of RNN
    batch_size = 32  # minibatch size
    seq_length = 30  # sequence length
    num_epochs = 50  # number of epochs
    learning_rate = 0.001  # learning rate
    sequences_step = 1  # step to create sequences

    md = bidirectional_lstm_model(seq_length, vocab_size)
    md.summary()

    # fit the model
    callbacks = [EarlyStopping(patience=4, monitor='val_loss'),
                 ModelCheckpoint(
                     filepath=save_dir +'save_model' +'my_model_gen_sentences_lstm.{epoch:02d}-{val_loss:.2f}.hdf5', \
                     monitor='val_loss', verbose=0, mode='auto', period=2)]
    history = md.fit(X, y,
                     batch_size=batch_size,
                     shuffle=True,
                     epochs=num_epochs,
                     callbacks=callbacks,
                     validation_split=0.01)

    # load vocabulary
    print("loading vocabulary...")
    vocab_file = os.path.join(save_dir, "words_vocab.pkl")

    with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

    vocab_size = len(words)

    from keras.models import load_model

    # load the model
    print("loading model...")
    input_file =glob.glob(os.path.join(save_dir+'*.hdf5'))
    print(input_file)
    # model = load_model(save_dir +'my_model_gen_sentences_lstm.final.hdf5')
    #
    #
    # def sample(preds, temperature=1.0):
    #     # helper function to sample an index from a probability array
    #     preds = np.asarray(preds).astype('float64')
    #     preds = np.log(preds) / temperature
    #     exp_preds = np.exp(preds)
    #     preds = exp_preds / np.sum(exp_preds)
    #     probas = np.random.multinomial(1, preds, 1)
    #     return np.argmax(probas)
    #
    #
    # seed_sentences = "What are your hobbies"
    # generated = ''
    # sentence = []
    # for i in range(seq_length):
    #     sentence.append("a")
    #
    # seed = seed_sentences.split()
    #
    # for i in range(len(seed)):
    #     sentence[seq_length - i - 1] = seed[len(seed) - i - 1]
    #
    # generated += ' '.join(sentence)
    # print('Generating text with the following seed: "' + ' '.join(sentence) + '"')
    #
    # print()
