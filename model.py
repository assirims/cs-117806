# ==================== model.py ====================
# BiRNN classifier

import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Flatten, Dense, Bidirectional, SimpleRNN, Input
from tensorflow.keras.models import Model
from config import SEQUENCE_LENGTH


def build_birnn(input_shape, num_classes):
    inp = Input(shape=input_shape)  # (seq_length, H*W*channels after feature extractor)
    x = Bidirectional(SimpleRNN(128, return_sequences=True))(inp)
    x = Bidirectional(SimpleRNN(64))(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out, name='birnn')

