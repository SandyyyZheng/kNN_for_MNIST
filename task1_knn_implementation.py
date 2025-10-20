# Task 1: Basic kNN implementation and testing
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist_data, kNN, visualize_digit, analyze_predictions


def main():
    print("="*60)
    print("Task 1: kNN Implementation and Basic Testing")
    print("="*60)
    
    # load the MNIST dataset
    print("Loading MNIST data...")
    images_train, images_test, labels_train, labels_test = load_mnist_data()
    
    print(f"Training data shape: {images_train.shape}")
    print(f"Test data shape: {images_test.shape}")
    print(f"Training labels shape: {labels_train.shape}")
    print(f"Test labels shape: {labels_test.shape}")
    
    # check one of the training images
    print("\nDisplaying sample training image...")
    sample_idx = 10
    sample_image = images_train[sample_idx, :]
    sample_label = labels_train[sample_idx][0]
    visualize_digit(sample_image, sample_label)
    
    # Now test my kNN implementation with k=1
    print("\nTesting kNN with k=1...")
    acc, acc_av = kNN(images_train, labels_train, images_test, labels_test, k=1)
    
    print("\nResults:")
    print("Accuracy for each class (0-9):")
    for i in range(10):
        print(f"Class {i}: {acc[i]:.4f}")
    print(f"\nAverage accuracy: {acc_av:.4f}")
    
    # right and wrong examples
    analyze_predictions(images_train, labels_train, images_test, labels_test)
    
    print("\n" + "="*60)
    print("Task 1 completed!")
    print("="*60)


if __name__ == "__main__":
    main()