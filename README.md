# k-Nearest Neighbors (kNN) for MNIST Digit Classification

This project implements k-Nearest Neighbors algorithm for handwritten digit classification using the MNIST dataset.

## File Structure

```
kNN/
├── utils.py    # Shared utility functions (data loading, kNN implementation)
├── task1_knn_implementation.py
├── task2_training_size_effect.py
├── task3_k_value_effect.py
├── task4_cross_validation.py
├── requirements.txt
├── README.md
└── MNIST_digit_data.mat    # MNIST dataset (needs to be downloaded)
```

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download the `MNIST_digit_data.mat` file [here](http://yann.lecun.com/exdb/mnist/)
2. Place the file in the `kNN_for_MNIST/` directory

## Usage

### Run Individual Tasks
```bash
# Task 1: kNN Implementation
python task1_knn_implementation.py

# Task 2: Training Size Effect
python task2_training_size_effect.py

# Task 3: k Value Effect  
python task3_k_value_effect.py

# Task 4: Cross-Validation
python task4_cross_validation.py
```

## Task Description

Implement kNN in Python for handwritten digit classification and submit all codes and plots:
Download MNIST digit dataset (60,000 training and 10,000 testing data points) and the starter code from the course page. Each row in the matrix represents a handwritten digit image. The starter code shows how to visualize an example data point. The task is to predict the class (0 to 9) for a given test image, so it is a 10-way classification problem.

### Task 1: kNN Function Implementation 

Write a Python function that implements **kNN** for this task and reports the accuracy for each class (10 numbers) as well as the average accuracy (one number).

_[acc acc_av] = kNN(images_train, labels_train, images_test, labels_test, k)_

where _acc_ is a vector of length 10 and _acc_av_ is a scalar. Look at a few correct and wrong predictions to see if it makes sense. To speed it up, in all experiments, you may use only the first 1000 testing images.

### Task 2: Effect of Training Data Size on Performance

For **k = 1**, change the number of training data points (30 to 10,000) to see the change in performance. Plot the average accuracy for 10 different dataset sizes. In the plot, the x-axis is for the number of training data and the y-axis is for the accuracy.

### Task 3: Effect of k Value on Accuracy

Show the effect of **k** on the accuracy. Make a plot similar to the above one with multiple colored curves on top of each other (each for a particular _k_ in `[1, 3, 5, 10]`).

### Task 4: Cross-Validation for Best k Selection

First choose 2,000 training data randomly (to speed up the experiment). Then, split the training data randomly into two halves (the first for training and the second for cross-validation to choose the best _k_). Please plot the average accuracy with respect to _k_ on the validation set. You may search for _k_ in this list: `[1, 3, 5, 10]`. Finally, report the accuracy for the best _k_ on the testing data.

## 📖 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

