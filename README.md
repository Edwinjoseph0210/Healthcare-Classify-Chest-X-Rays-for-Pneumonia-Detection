# 🏥 Pneumonia Detection from Chest X-Rays

## Project Overview
This project uses a **Convolutional Neural Network (CNN)** to classify **chest X-ray images** and detect pneumonia.  
AI-based medical imaging tools like this can assist doctors in early diagnosis and treatment of lung diseases.

## Key Features
- Classifies chest X-rays as pneumonia / normal.
- Simple CNN architecture.
- Fast training and evaluation.

## Requirements
- Python 3.x
- TensorFlow / Keras
- pandas
- numpy
- matplotlib

## Usage
1. Download or prepare a chest X-ray dataset (labelled `pneumonia` and `normal`).
2. Run `pneumonia_detection.py` to train the model.
3. Evaluate model accuracy and visualize predictions.

## Example Output
Model accuracy: 93%  
Sample classified images displayed in plot.

## Future Work
- Integrate transfer learning (ResNet, EfficientNet).
- Improve model explainability (Grad-CAM visualizations).
- Deploy as an online diagnostic tool.
