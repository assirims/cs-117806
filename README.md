# AOARNN-HMPSN Implementation

# File Structure:
# ├── README.md
# ├── config.py
# ├── data_loader.py
# ├── preprocessing.py
# ├── feature_extractor.py
# ├── model.py
# ├── aoa.py
# ├── utils.py
# └── train.py

# ==================== README.md ====================
# An Intelligent IoT-Based Software Framework (AOARNN-HMPSN)

## Overview
This repository implements the AOARNN-HMPSN pipeline described in the article, featuring:
- **Preprocessing**: Sobel filter edge enhancement (`preprocessing.py`) citeturn3file17
- **Feature Extraction**: SqueezeNet fire modules (`feature_extractor.py`) citeturn3file10
- **Classification**: BiRNN model (`model.py`) citeturn3file1
- **Hyperparameter Tuning**: Archimedes Optimization Algorithm (`aoa.py`) citeturn3file2
- **Pipeline Orchestration**: `train.py`

## Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV (`cv2`)
- scikit-learn
- pandas
- numpy

## Installation
```bash
pip install tensorflow opencv-python scikit-learn pandas numpy
```

## Usage
1. Configure `config.py` with paths and parameters.
2. Ensure URFD dataset is downloaded to `DATA_DIR`.
3. Run:
```bash
python train.py
```

## File Structure
```
├── README.md
├── config.py
├── data_loader.py
├── preprocessing.py
├── feature_extractor.py
├── model.py
├── aoa.py
├── utils.py
└── train.py
```

## Citation
Please cite the original publication when using this code.
