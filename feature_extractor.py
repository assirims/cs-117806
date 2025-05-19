# ==================== feature_extractor.py ====================
# SqueezeNet implementation (Fire module)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, concatenate, MaxPooling2D, Input
from tensorflow.keras.models import Model
from config import SQUEEZENET_INPUT_SHAPE, FIRE_SQUEEZE, FIRE_EXPAND


def fire_module(x, squeeze_filters, expand_filters):
    sq = Conv2D(squeeze_filters, (1,1), activation='relu', padding='same')(x)
    ex1 = Conv2D(expand_filters, (1,1), activation='relu', padding='same')(sq)
    ex3 = Conv2D(expand_filters, (3,3), activation='relu', padding='same')(sq)
    return concatenate([ex1, ex3], axis=-1)


def build_squeezenet(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2,2))(x)
    # Five fire modules with interleaved pooling
    for i in range(5):
        x = fire_module(x, FIRE_SQUEEZE, FIRE_EXPAND)
        if i % 2 == 1:
            x = MaxPooling2D((2,2))(x)
    model = Model(inp, x, name='squeezenet')
    return model