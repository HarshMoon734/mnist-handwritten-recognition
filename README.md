# MNIST Handwritten Digit Classifier

This repository contains a machine learning model built to classify handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN). The model is implemented using TensorFlow and Keras. The main functionality of the project is demonstrated in the `main.ipynb` file.

## Features

- **Data Loading & Preprocessing:** The MNIST dataset is loaded and normalized for input to the CNN.
- **CNN Model Architecture:** The model consists of multiple convolutional layers followed by max-pooling layers and dense layers. The final output layer uses softmax for multi-class classification.
- **Model Training & Evaluation:** The model is trained on the MNIST dataset and evaluated for accuracy on the test set. The training process also includes validation.
- **Model Saving:** After training, the model is saved as a `.keras` file for later use.

## Requirements

To run this project, you need to install the following libraries:

- TensorFlow
- OpenCV
- Numpy
- scikit-learn

You can install them using pip:

```bash
pip install tensorflow opencv-python numpy scikit-learn
