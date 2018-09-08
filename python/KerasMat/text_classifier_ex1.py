import os

from keras.preprocessing.text import Tokenizer
from konlpy.tag import Okt
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Model
from scipy.sparse import csr_matrix

import numpy as np


def morpheme_analyze(texts):
    twitter = Okt()
    the_list = []
    for text in texts:
        pos_list = twitter.pos(text, "utf-8")
        words = []
        for p in pos_list:
            if p[1] == 'Josa' or p[1] == 'Eomi' or p[1] == 'Punctuation':
                continue
            words.append(p[0])
        the_list.append(" ".join(words))
    return the_list


def read_corpus(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    corpus_x = []
    corpus_y = []
    for f in files:
        with open(os.path.join(path, f)) as fp:
            for line in fp:
                splits = line.strip().split("\t")
                if len(splits) < 3:
                    continue
                # title, contents
                corpus_x.append(" ".join(splits[2:4]))
                corpus_y.append(splits[1])
        fp.close()

    np.random.seed(100)
    shuffle_indices = np.random.permutation(np.arange(len(corpus_x)))
    x_shuffled = np.array(corpus_x)[shuffle_indices]
    y_shuffled = np.array(corpus_y)[shuffle_indices]

    return x_shuffled, y_shuffled


if __name__ == "__main__":
    texts, tags = read_corpus("../../corpus")
    texts = morpheme_analyze(texts)

    train_size = int(len(texts) * 0.8)
    train_texts = texts[:train_size]
    train_tags = tags[:train_size]
    test_texts = texts[train_size:]
    test_tags = tags[train_size:]

    num_labels = len(set(train_tags))
    vocab_size = 10000
    batch_size = 100

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_texts)

    x_train = csr_matrix(tokenizer.texts_to_matrix(train_texts, mode="count")).todense()
    x_test = csr_matrix(tokenizer.texts_to_matrix(test_texts, mode="count")).todense()

    encoder = LabelBinarizer()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    hidden1 = Dense(512)
    hidden2 = Dense(256)
    output = Dense(num_labels)
    relu = Activation("relu")
    softmax = Activation("softmax")
    dropout = Dropout(0.3)

    x = Input(shape=(vocab_size,))
    h1 = dropout(relu(hidden1(x)))
    h2 = dropout(relu(hidden2(h1)))
    y = softmax(softmax(output(h2)))
    model = Model(x, y)

    # model = Sequential()
    # model.add(Dense(512, input_shape=(vocab_size,)))
    # model.add(Activation("relu"))
    # model.add(Dropout(0.3))
    # model.add(Dense(num_labels))
    # model.add(Activation("softmax"))
    # model.summary()
    #
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=30,
                        verbose=1,
                        validation_split=0.1)

    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)

    print("Test Accuracy: ", score[1])

    predictions = model.predict(x_test)

    text_labels = encoder.classes_
    for i in range(len(test_texts)):
        print(test_texts[i])
        print("Actual label:", test_tags[i])
        print("Predict label:", text_labels[np.argmax(predictions[i])])