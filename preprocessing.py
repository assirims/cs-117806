# ==================== preprocessing.py ====================
# Apply Sobel filter to enhance edges

import numpy as np
import cv2

def apply_sobel(X):
    """
    Input X: (N, seq_length, H, W)
    Output: (N, seq_length, H, W, 1)
    """
    N, L, H, W = X.shape
    X_sobel = np.zeros((N, L, H, W, 1), dtype=np.float32)
    for i in range(N):
        for t in range(L):
            sobelx = cv2.Sobel(X[i,t], cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(X[i,t], cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(sobelx, sobely)
            X_sobel[i,t,...,0] = mag / np.max(mag)
    return X_sobel
