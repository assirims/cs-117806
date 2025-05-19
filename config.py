# ==================== config.py ====================
# Configuration parameters

# Path to pre-downloaded URFD dataset (videos folder)
DATA_DIR = "/path/to/URFD"

# Train/Test split (70/30)
TRAIN_RATIO = 0.7
RANDOM_STATE = 42

# AOA parameters citeturn3file2
AOA_POP_SIZE = 30
AOA_MAX_ITER = 50
AOA_LOWER = 0.0  # lower bound for solutions
AOA_UPPER = 1.0  # upper bound for solutions

# Model parameters
SEQUENCE_LENGTH = 30    # number of frames per sample
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 25

# Feature extractor (SqueezeNet) parameters citeturn3file10
SQUEEZENET_INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
FIRE_SQUEEZE = 16
FIRE_EXPAND = 64