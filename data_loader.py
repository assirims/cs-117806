# ==================== data_loader.py ====================
# Load video sequences and labels

import os
import cv2
import numpy as np

def load_urfd(dataset_dir, seq_length, img_h, img_w):
    """
    Traverse URFD directory structure and load sequences of Sobel-filtered frames.
    Assumes subfolders 'Fall' and 'NoFall'.
    Returns X: array of shape (N, seq_length, img_h, img_w, 1), y: labels.
    """
    X, y = [], []
    for label, cls in enumerate(['NoFall', 'Fall']):
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.mp4','.avi')):
                continue
            path = os.path.join(cls_dir, fname)
            cap = cv2.VideoCapture(path)
            frames = []
            while len(frames) < seq_length:
                ret, frame = cap.read()
                if not ret:
                    break
                # resize and grayscale
                frame = cv2.resize(frame, (img_w, img_h))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            cap.release()
            if len(frames) == seq_length:
                X.append(np.stack(frames, axis=0))
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y
