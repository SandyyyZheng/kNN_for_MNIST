# Task 3: Does the value of k matter?
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist_data, kNN


def main():
    print("="*60)
    print("Task 3: Effect of k Value on Accuracy")
    print("="*60)
    
    print("Loading MNIST data...")
    images_train, images_test, labels_train, labels_test = load_mnist_data()
    
    # Test different k values
    k_values = [1, 3, 5, 10]
    colors = ['blue', 'red', 'green', 'orange']
    training_sizes_k = [30, 100, 300, 500, 1000, 2000, 3000, 5000]
    
    print(f"Testing {len(k_values)} different k values: {k_values}")
    print(f"Across {len(training_sizes_k)} training sizes: {training_sizes_k}")
    
    plt.figure(figsize=(14, 10))
    
    # Keep track of all results
    all_results = {}
    
    for i, k in enumerate(k_values):
        accuracies_k = []
        print(f"\nTesting k={k}...")
        
        for size in training_sizes_k:
            print(f"  Training size: {size}")
            
            train_subset = images_train[:size]
            labels_subset = labels_train[:size]
            
            acc, acc_av = kNN(train_subset, labels_subset, images_test, labels_test, k=k)
            accuracies_k.append(acc_av)
        
        all_results[k] = accuracies_k
        
        # Plot this k's performance curve
        plt.plot(training_sizes_k, accuracies_k, 'o-', 
                color=colors[i], linewidth=2, markersize=6, label=f'k={k}')
        
        print(f"k={k} accuracies: {[f'{acc:.4f}' for acc in accuracies_k]}")
    
    plt.xlabel('Number of Training Data Points', fontsize=12)
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.title('kNN Performance vs Training Data Size for Different k Values', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.minorticks_on()
    plt.grid(True, which='minor', alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    # Detailed analysis of each k's performance
    print("\n" + "="*60)
    print("Detailed Analysis:")
    print("="*60)
    
    for j, size in enumerate(training_sizes_k):
        accuracies_at_size = [all_results[k][j] for k in k_values]
        best_k_idx = np.argmax(accuracies_at_size)
        best_k = k_values[best_k_idx]
        best_acc = accuracies_at_size[best_k_idx]
        
        print(f"Training size {size:4d}: Best k={best_k} (accuracy: {best_acc:.4f})")
    
    print("\n" + "="*60)
    print("Overall Best Performance:")
    print("="*60)
    
    best_overall_acc = 0
    best_overall_k = 0
    best_overall_size = 0
    
    for k in k_values:
        for j, size in enumerate(training_sizes_k):
            acc = all_results[k][j]
            if acc > best_overall_acc:
                best_overall_acc = acc
                best_overall_k = k
                best_overall_size = size
    
    print(f"Best overall: k={best_overall_k}, training_size={best_overall_size}, accuracy={best_overall_acc:.4f}")
    
    # How consistent is each k value
    print("\n" + "="*60)
    print("k Value Trends:")
    print("="*60)
    
    for k in k_values:
        accuracies = all_results[k]
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"k={k}: Average accuracy={avg_acc:.4f}, Std={std_acc:.4f}")
    
    print("\n" + "="*60)
    print("Task 3 completed!")
    print("="*60)


if __name__ == "__main__":
    main()