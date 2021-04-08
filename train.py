from time import time

from keras import backend, optimizers
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, Flatten, Input, Lambda, Layer, \
    LSTM, Permute, RepeatVector, TimeDistributed
from keras.layers.merge import multiply, concatenate
from keras.models import Model

from params import BATCH_SIZE, EMBEDDING_DIM, EPOCHS, HIDDEN_LAYERS, MAX_SEQ_LENGTH


def train(x_train, y_train, x_validation, y_validation, embeddings):
    left_input = Input(shape=(MAX_SEQ_LENGTH,), dtype="float32")
    right_input = Input(shape=(MAX_SEQ_LENGTH,), dtype="float32")
    left_sen_representation = _shared_model(left_input, embeddings)
    right_sen_representation = _shared_model(right_input, embeddings)

    man_distance = ManDist()(
        [left_sen_representation, right_sen_representation])
    sen_representation = concatenate(
        [left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation="sigmoid")(
        Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss="mean_squared_error",
                  optimizer=optimizers.Adam(), metrics=["accuracy"])
    model.summary()

    training_start_time = time()
    malstm_trained = model.fit([x_train["left"], x_train["right"]],
                               y_train,
                               batch_size=BATCH_SIZE,
                               epochs=EPOCHS,
                               validation_data=([x_validation["left"], x_validation["right"]], y_validation))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" %
          (EPOCHS, training_end_time - training_start_time))

    return model


class ManDist(Layer):
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    def call(self, x, **kwargs):
        self.result = backend.exp(-backend.sum(
            backend.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    def compute_output_shape(self, input_shape):
        return backend.int_shape(self.result)


def _shared_model(input, embeddings):
    embedded = Embedding(len(embeddings), EMBEDDING_DIM, weights=[embeddings], input_shape=(MAX_SEQ_LENGTH,),
                         trainable=False)(input)

    activations = Bidirectional(
        LSTM(HIDDEN_LAYERS, return_sequences=True), merge_mode="concat")(embedded)
    activations = Bidirectional(
        LSTM(HIDDEN_LAYERS, return_sequences=True), merge_mode="concat")(activations)

    activations = Dropout(0.5)(activations)

    attention = TimeDistributed(Dense(1, activation="tanh"))(activations)
    attention = Flatten()(attention)
    attention = Activation("softmax")(attention)
    attention = RepeatVector(HIDDEN_LAYERS * 2)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([activations, attention])
    sent_representation = Lambda(
        lambda xin: backend.sum(xin, axis=1))(sent_representation)

    sent_representation = Dropout(0.1)(sent_representation)

    return sent_representation
