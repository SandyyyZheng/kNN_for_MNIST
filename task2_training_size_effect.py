# Task 2: How does training data size affect performance?
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist_data, kNN


def main():
    print("="*60)
    print("Task 2: Effect of Training Data Size on Performance")
    print("="*60)
    
    print("Loading MNIST data...")
    images_train, images_test, labels_train, labels_test = load_mnist_data()
    
    # Try different amounts of training data
    training_sizes = [30, 100, 300, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    accuracies_vs_size = []
    
    print(f"Testing {len(training_sizes)} different training sizes with k=1...")
    
    for size in training_sizes:
        print(f"\nTesting with {size} training samples...")
        
        train_subset = images_train[:size]
        labels_subset = labels_train[:size]
        
        acc, acc_av = kNN(train_subset, labels_subset, images_test, labels_test, k=1)
        accuracies_vs_size.append(acc_av)
        print(f"Average accuracy with {size} training samples: {acc_av:.4f}")
    
    # plot
    plt.figure(figsize=(12, 8))
    plt.plot(training_sizes, accuracies_vs_size, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Training Data Points', fontsize=12)
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.title('kNN Performance vs Training Data Size (k=1)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add the accuracy values on the plot
    for i, (size, acc) in enumerate(zip(training_sizes, accuracies_vs_size)):
        plt.annotate(f'{acc:.3f}', (size, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("\n" + "="*60)
    print("Summary Results:")
    print("="*60)
    print(f"Training sizes: {training_sizes}")
    print(f"Accuracies: {[f'{acc:.4f}' for acc in accuracies_vs_size]}")
    
    best_idx = np.argmax(accuracies_vs_size)
    worst_idx = np.argmin(accuracies_vs_size)
    
    print(f"\nBest performance: {accuracies_vs_size[best_idx]:.4f} with {training_sizes[best_idx]} training samples")
    print(f"Worst performance: {accuracies_vs_size[worst_idx]:.4f} with {training_sizes[worst_idx]} training samples")
    print(f"Improvement: {accuracies_vs_size[best_idx] - accuracies_vs_size[worst_idx]:.4f}")
    
    print("\n" + "="*60)
    print("Task 2 completed!")
    print("="*60)


if __name__ == "__main__":
    main()