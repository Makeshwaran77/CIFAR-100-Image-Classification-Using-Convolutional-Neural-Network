Project Overview

This project focuses on multi-class image classification using the CIFAR-100 dataset and a Convolutional Neural Network (CNN). CIFAR-100 is a challenging benchmark dataset consisting of 100 fine-grained object categories, making it well-suited for evaluating deep learning models for visual recognition tasks.

The project includes exploratory data analysis (EDA), data preprocessing, CNN model design, training, and performance evaluation. The goal is to demonstrate how convolutional neural networks effectively learn spatial features from images and perform large-scale image classification.

Dataset Description

CIFAR-100 contains:

60,000 color images of size 32×32

100 fine-grained classes

500 images per class

50,000 training images

10,000 testing images

Each image belongs to exactly one fine label category, making it a balanced and complex dataset.

Exploratory Data Analysis (EDA)

The EDA phase includes:

Visualization of random image samples from different classes

Verification of class balance across all 100 categories

Pixel intensity distribution analysis

Inspection of image dimensions and data structure

EDA confirms uniform class distribution and consistent image dimensions across the dataset.

Data Preprocessing

The following preprocessing steps are applied:

Pixel values normalized to the range [0, 1]

Labels converted into one-hot encoded vectors

Data prepared in a format suitable for CNN input

Model Architecture

The Convolutional Neural Network consists of:

Multiple convolutional layers with ReLU activation

Max pooling layers for spatial downsampling

Batch normalization for training stability

Fully connected dense layers for classification

Dropout for regularization

Softmax output layer with 100 units

The model is trained using the Adam optimizer and categorical cross-entropy loss function.

Model Training

Training performed with a validation split

Mini-batch gradient descent used for optimization

Accuracy and validation accuracy monitored across epochs

Evaluation Metrics

Training Accuracy

Validation Accuracy

Test Accuracy

Accuracy vs Validation Accuracy performance graph

The baseline CNN model achieves approximately 45–55% test accuracy, which can be improved with deeper architectures or transfer learning.

Results and Observations

CNN effectively captures spatial features from images

CIFAR-100 presents higher complexity compared to CIFAR-10 due to fine-grained classes

Regularization techniques help reduce overfitting

Performance validates the suitability of CNNs for image classification tasks

Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

Google Colab

Future Enhancements

Apply data augmentation techniques

Use pretrained models such as ResNet or EfficientNet

Experiment with deeper CNN architectures

Perform hierarchical classification using coarse labels

Author

Makeshwaran
Third Year AI & ML Student
