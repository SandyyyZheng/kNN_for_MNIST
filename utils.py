# Helper functions for kNN digit classification
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_mnist_data(filename='MNIST_digit_data.mat', seed=3):
    # Load the .mat file
    M = loadmat(filename)
    images_train = M['images_train']
    images_test = M['images_test'] 
    labels_train = M['labels_train']
    labels_test = M['labels_test']
    
    # Shuffle the data randomly so I don't get biased results
    np.random.seed(seed)
    
    inds = np.random.permutation(images_train.shape[0])
    images_train = images_train[inds]
    labels_train = labels_train[inds]
    
    inds = np.random.permutation(images_test.shape[0])
    images_test = images_test[inds]
    labels_test = labels_test[inds]
    
    return images_train, images_test, labels_train, labels_test


def kNN(images_train, labels_train, images_test, labels_test, k, max_test_samples=1000):
    # Only use first 1000 test images to make it run faster
    images_test = images_test[:max_test_samples]
    labels_test = labels_test[:max_test_samples]
    
    predictions = []
    
    print(f"Running kNN with k={k}...")
    
    for i in range(len(images_test)):
        if i % 100 == 0:
            print(f"Processing test image {i}/{len(images_test)}")
        
        test_image = images_test[i]
        # Calculate euclidean distance to every training image
        distances = np.sqrt(np.sum((images_train - test_image)**2, axis=1))
        
        # Get the k closest training images
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = labels_train[k_nearest_indices].flatten()
        
        # vote & pick the most common label among the k neighbors
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predictions.append(predicted_label)
    
    predictions = np.array(predictions)
    true_labels = labels_test.flatten()
    
    # Calculate how well it did for each digit class
    acc = np.zeros(10)
    for class_label in range(10):
        class_mask = (true_labels == class_label)
        if np.sum(class_mask) > 0:
            class_predictions = predictions[class_mask]
            acc[class_label] = np.mean(class_predictions == class_label)
    
    # overall accuracy
    acc_av = np.mean(predictions == true_labels)
    
    return acc, acc_av


def visualize_digit(image, label, title_prefix="Class Label"):
    # reshape the flattened image back to 28x28 and show it
    im = image.reshape((28, 28), order='F')
    plt.imshow(im, cmap='gray')
    plt.title(f'{title_prefix}: {label}')
    plt.axis('off')
    plt.show()


def analyze_predictions(images_train, labels_train, images_test, labels_test, num_samples=100):
    print("Analyzing some predictions...")
    test_subset = images_test[:num_samples]
    labels_subset = labels_test[:num_samples].flatten()
    
    # some examples
    predictions_subset = []
    for i in range(num_samples):
        test_image = test_subset[i]
        distances = np.sqrt(np.sum((images_train - test_image)**2, axis=1))
        nearest_idx = np.argmin(distances)
        predicted_label = labels_train[nearest_idx][0]
        predictions_subset.append(predicted_label)
    
    predictions_subset = np.array(predictions_subset)
    
    # correct
    correct_mask = (predictions_subset == labels_subset)
    correct_indices = np.where(correct_mask)[0][:3]
    
    print(f"\nSome correct predictions:")
    for idx in correct_indices:
        print(f"Test image {idx}: True={labels_subset[idx]}, Predicted={predictions_subset[idx]}")
    
    # wrong
    wrong_indices = np.where(~correct_mask)[0][:3]
    print(f"\nSome wrong predictions:")
    for idx in wrong_indices:
        print(f"Test image {idx}: True={labels_subset[idx]}, Predicted={predictions_subset[idx]}")