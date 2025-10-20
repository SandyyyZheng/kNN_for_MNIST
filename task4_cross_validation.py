# Task 4: Use cross-validation to pick the best k
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist_data, kNN


def main():
    print("="*60)
    print("Task 4: Cross-Validation for Best k Selection")
    print("="*60)
    
    print("Loading MNIST data...")
    images_train, images_test, labels_train, labels_test = load_mnist_data()
    
    # pick 2000 random training samples
    print("Selecting 2000 training samples randomly...")
    np.random.seed(3)
    random_indices = np.random.choice(len(images_train), 2000, replace=False)
    selected_train_images = images_train[random_indices]
    selected_train_labels = labels_train[random_indices]
    
    # Split: first 1000 for training, next 1000 for validation
    train_images_cv = selected_train_images[:1000]
    train_labels_cv = selected_train_labels[:1000]
    val_images_cv = selected_train_images[1000:]
    val_labels_cv = selected_train_labels[1000:]
    
    print(f"Cross-validation setup:")
    print(f"  Training set size: {len(train_images_cv)}")
    print(f"  Validation set size: {len(val_images_cv)}")
    print(f"  Test set size: {len(images_test)} (will use first 1000)")
    
    # Try different k values and see which works best on validation set
    k_candidates = [1, 3, 5, 10]
    validation_accuracies = []
    validation_class_accuracies = []
    
    print(f"\nTesting k candidates: {k_candidates}")
    print("Running cross-validation...")
    
    for k in k_candidates:
        print(f"\nValidating k={k}...")
        
        # Train on training set, test on validation set
        acc, acc_av = kNN(train_images_cv, train_labels_cv, val_images_cv, val_labels_cv, k=k)
        validation_accuracies.append(acc_av)
        validation_class_accuracies.append(acc)
        print(f"Validation accuracy for k={k}: {acc_av:.4f}")
    
    # pick the k that did best on validation
    best_k_idx = np.argmax(validation_accuracies)
    best_k = k_candidates[best_k_idx]
    best_val_acc = validation_accuracies[best_k_idx]
    
    print("\n" + "="*40)
    print("Cross-Validation Results:")
    print("="*40)
    for i, k in enumerate(k_candidates):
        marker = " <-- BEST" if i == best_k_idx else ""
        print(f"k={k}: {validation_accuracies[i]:.4f}{marker}")
    
    print(f"\nSelected best k: {best_k} (validation accuracy: {best_val_acc:.4f})")
    
    # plots
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(k_candidates)), validation_accuracies, 
                   color=['red' if i == best_k_idx else 'lightblue' for i in range(len(k_candidates))],
                   alpha=0.7, edgecolor='black')
    
    # bar chart
    for i, (bar, acc) in enumerate(zip(bars, validation_accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('k Value', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Cross-Validation: Accuracy vs k Value', fontsize=14)
    plt.xticks(range(len(k_candidates)), k_candidates)
    plt.ylim(0, max(validation_accuracies) * 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # line chart
    plt.figure(figsize=(10, 6))
    plt.plot(k_candidates, validation_accuracies, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('k Value', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Cross-Validation: Accuracy vs k Value (Line Plot)', fontsize=14)
    plt.xticks(k_candidates)
    plt.grid(True, alpha=0.3)
    plt.ylim(min(validation_accuracies) * 0.95, max(validation_accuracies) * 1.05)
    
    # Highlight the winner
    plt.scatter([best_k], [best_val_acc], color='red', s=100, zorder=5)
    plt.annotate(f'Best k={best_k}\nAcc={best_val_acc:.4f}', 
                xy=(best_k, best_val_acc), xytext=(best_k, best_val_acc + 0.01),
                ha='center', fontweight='bold', 
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.show()
    
    # test the best k on the real test set
    print("\n" + "="*40)
    print("Final Test Evaluation:")
    print("="*40)
    print(f"Testing best k={best_k} on test set using all 2000 selected training samples...")
    
    final_acc, final_acc_av = kNN(selected_train_images, selected_train_labels, 
                                  images_test, labels_test, k=best_k)
    
    print(f"\nFinal test results with k={best_k}:")
    print("Per-class accuracy (0-9):")
    for i in range(10):
        print(f"  Class {i}: {final_acc[i]:.4f}")
    print(f"\nFinal average test accuracy: {final_acc_av:.4f}")
    
    # Performance compared to validation
    print("\n" + "="*40)
    print("Performance Comparison:")
    print("="*40)
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy:       {final_acc_av:.4f}")
    print(f"Difference:          {final_acc_av - best_val_acc:.4f}")
    
    if final_acc_av > best_val_acc:
        print("✓ Test performance is better than validation (good generalization)")
    elif abs(final_acc_av - best_val_acc) < 0.01:
        print("✓ Test performance is similar to validation (stable model)")
    else:
        print("⚠ Test performance is lower than validation (possible overfitting)")
    
    # summary
    print("\n" + "="*40)
    print("Summary Statistics:")
    print("="*40)
    print(f"Best k value: {best_k}")
    print(f"Training samples used: 2000")
    print(f"Validation samples used: 1000") 
    print(f"Test samples used: 1000")
    print(f"Final test accuracy: {final_acc_av:.4f}")
    print(f"Best class accuracy: {np.max(final_acc):.4f} (class {np.argmax(final_acc)})")
    print(f"Worst class accuracy: {np.min(final_acc):.4f} (class {np.argmin(final_acc)})")
    
    print("\n" + "="*60)
    print("Task 4 completed!")
    print("="*60)


if __name__ == "__main__":
    main()