import pickle
import re

from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from numpy.random import randn
from pandas import read_csv

from params import EMBEDDING_DIM, MAX_SEQ_LENGTH


def embed_for_training(train_data_path, validation_data_path, embedding_path):
    train_df = read_csv(train_data_path)
    train_df["Question_1_n"] = [[] for _ in range(len(train_df.index))]
    train_df["Question_2_n"] = [[] for _ in range(len(train_df.index))]

    validation_df = read_csv(validation_data_path)
    validation_df["Question_1_n"] = [[]
                                     for _ in range(len(validation_df.index))]
    validation_df["Question_2_n"] = [[]
                                     for _ in range(len(validation_df.index))]

    embedding_dict = KeyedVectors.load_word2vec_format(
        embedding_path, binary=True)

    train_size = len(train_df)
    df = train_df.append(validation_df, ignore_index=True)
    df, embeddings, vocabs = _make_w2v_embeddings(embedding_dict, df)
    train_df = df[:train_size]
    validation_df = df[train_size:]

    _dump_data(train_df.loc[train_df["flag"] == 1], vocabs)

    x_train = train_df[["Question_1_n", "Question_2_n"]]
    y_train = train_df["flag"]

    x_validation = validation_df[["Question_1_n", "Question_2_n"]]
    y_validation = validation_df["flag"]

    x_train = _split_and_zero_padding(x_train)
    x_validation = _split_and_zero_padding(x_validation)

    y_train = y_train.values
    y_validation = y_validation.values

    assert x_train["left"].shape == x_train["right"].shape
    assert len(x_train["left"]) == len(y_train)

    return x_train, y_train, x_validation, y_validation, embeddings


def embed_for_serving():
    df, vocabs = _load_data()

    candidates = []
    for index, row in df.iterrows():
        candidates.append(row["Question_1"])

    return df, vocabs, candidates


def embed_for_request(df, vocabs, question):
    df = df.copy()

    q2n = []
    words = _text_to_word_list(question)
    for word in words:
        if word in vocabs:
            q2n.append(vocabs[word])

    for index, _ in df.iterrows():
        df.at[index, "Question_2_n"] = q2n

    X = _split_and_zero_padding(df)
    return X["left"], X["right"]


def _make_w2v_embeddings(word2vec, df):
    vocabs = {}
    vocabs_cnt = 0

    for index, row in df.iterrows():
        if index != 0 and index % 1000 == 0:
            print(str(index) + " sentences embedded.")

        for question in ["Question_1", "Question_2"]:
            q2n = []
            words = _text_to_word_list(row[question])

            for word in words:
                if word not in word2vec:
                    continue
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])
            df.at[index, question + "_n"] = q2n

    embeddings = 1 * randn(len(vocabs) + 1, EMBEDDING_DIM)
    embeddings[0] = 0

    for index in vocabs:
        vocab_word = vocabs[index]
        if vocab_word in word2vec:
            embeddings[index] = word2vec[vocab_word]

    return df, embeddings, vocabs


def _text_to_word_list(text):
    text = str(text)
    text = text.lower()

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def _split_and_zero_padding(df):
    X = {"left": df["Question_1_n"], "right": df["Question_2_n"]}
    X["left"] = pad_sequences(X["left"], padding="pre",
                              truncating="post", maxlen=MAX_SEQ_LENGTH)
    X["right"] = pad_sequences(
        X["right"], padding="pre", truncating="post", maxlen=MAX_SEQ_LENGTH)

    return X


def _dump_data(df, vocabs):
    print("Serializing df")
    df_file = open("cache/df.pkl", "wb+")
    pickle.dump(df, df_file)
    df_file.close()

    print("Serializing vocabs")
    vocabs_file = open("cache/vocabs.pkl", "wb+")
    pickle.dump(vocabs, vocabs_file)
    vocabs_file.close()


def _load_data():
    print("Deserializing df")
    df_file = open("cache/df.pkl", "rb")
    df = pickle.load(df_file)
    df_file.close()

    print("Desirializing vocabs")
    vocabs_file = open("cache/vocabs.pkl", "rb")
    vocabs = pickle.load(vocabs_file)
    vocabs_file.close()

    return df, vocabs
