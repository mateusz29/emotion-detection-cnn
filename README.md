# Emotion Detection Using CNN

## Project Overview

This project implements an emotion detection system using Convolutional Neural Network (CNN). It leverages the FER2013 dataset, which consists of facial expression images, to classify emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The project includes training and testing scripts, data preparation, and a user interface for processing images to detect emotions. The model has achieved an accuracy of 63.74% on the test set.

## Features

- **Detecting Emotions**: Provides a graphical user interface for uploading images and detecting emotions.
- **Data Augmentation**: Applies data augmentation techniques to enhance model performance.
- **Model Evaluation**: Includes script to evaluate model performance on test set, and report accuracy.
- **Pre-trained Model**: Optionally use a pre-trained model for faster testing and experimentation.
- **User-Friendly Interface**: Simple and intuitive GUI for non-technical users to interact with the model.

## Technology Stack

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Libraries**: 
  - `torch` for deep learning
  - `PIL`, `OpenCV` for image processing
  - `tkinter`, `customtkinter` for the graphical user interface

## Dataset

The FER2013 dataset contains grayscale images of facial expressions, categorized into 7 emotion classes. Each image is 48x48 pixels. The dataset is divided into training, validation, and test sets.

## Model Architecture

The CNN model consists of:

- **Convolutional Layers**: Four layers with increasing filter sizes (32, 64, 128, 256) and ReLU activations.
- **Batch Normalization**: Applied after each convolutional layer.
- **MaxPooling**: Reduces spatial dimensions.
- **Fully Connected Layers**: Two layers with dropout regularization.
- **Output Layer**: 7 units corresponding to emotion classes.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mateusz29/emotion-detection-cnn.git
   cd emotion-detection-cnn
   ```
2. **Set Up a Virtual Environment**
    ```bash
    py -m venv venv
    venv\Scripts\activate
    ```
3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage Instructions

1. **Prepare the Data**
    ```bash
    py prepare_data.py
    ```
2. **Train the Model**
    ```bash
    py train_model.py
    ```
3. **Test the Model**
    ```bash
    py test_model.py
    ```
4. **Run the User Interface**
    ```bash
    py user_interface.py
    ```

## Example

- **Input**: 
![Input Image](assets/input_image.png)

- **Detected Face**:
![Detected Face](assets/detected_face.png)

- **Predicted Emotion**:
![Predicted Emotion](assets/predicted_emotion.png)

## Results

- **Model Accuracy**: Achieved 63.74% accuracy on the test set.
