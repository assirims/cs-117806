# ==================== train.py ====================
# Main pipeline orchestration

import numpy as np
from config import DATA_DIR, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, EPOCHS, BATCH_SIZE
from data_loader import load_urfd
from preprocessing import apply_sobel
from feature_extractor import build_squeezenet
from model import build_birnn
from aoa import aoa_optimize
from utils import split_data, evaluate_model
import tensorflow as tf


def main():
    # 1. Load raw sequences
    X_raw, y = load_urfd(DATA_DIR, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH)

    # 2. Preprocessing (Sobel)
    X_prep = apply_sobel(X_raw)

    # 3. Feature extraction per-frame
    s_model = build_squeezenet((IMG_HEIGHT, IMG_WIDTH, 1))
    # TimeDistributed feature extraction
    feat_seq = tf.keras.layers.TimeDistributed(s_model)(X_prep)
    # Flatten spatial dims
    N, L, H, W, C = feat_seq.shape
    X_feat = np.reshape(feat_seq, (N, L, H*W*C))

    # 4. Train/test split
    X_train, X_test, y_train, y_test = split_data(X_feat, y)

    # 5. Hyperparameter tuning with AOA
    def fitness(ind):
        # Example: interpret ind as learning_rate in [0,1]
        lr = 1e-4 + ind[0] * (1e-2 - 1e-4)
        model = build_birnn((L, H*W*C), num_classes=2)
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=3, batch_size=BATCH_SIZE, verbose=0,
                            validation_data=(X_test, y_test))
        return 1 - history.history['val_accuracy'][-1]

    best_solution = aoa_optimize(fitness, dim=1)
    print("Best AOA solution:", best_solution)

    # 6. Final training
    model = build_birnn((L, H*W*C), num_classes=2)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_test, y_test))
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()