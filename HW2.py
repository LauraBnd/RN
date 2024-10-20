import numpy as np
from torchvision.datasets import MNIST
import torch.utils.data as data_utils


# Load MNIST dataset from local files
def load_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=False,  # Avoid re-downloading
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(np.array(image))
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

train_X, train_Y = load_mnist(True)
test_X, test_Y = load_mnist(False)

train_X = train_X / 255.0
test_X = test_X / 255.0


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


train_Y = one_hot_encode(train_Y, 10)
test_Y = one_hot_encode(test_Y, 10)

np.random.seed(42)
input_size = 784  # Each MNIST image is 28x28
num_classes = 10
learning_rate = 0.1

W = np.random.randn(input_size, num_classes) * 0.01  # Weight matrix
b = np.zeros((1, num_classes))  # Bias vector

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return loss

def accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)

def train_perceptron(train_X, train_Y, W, b, epochs=200, batch_size=100):  # 200 epochs
    m = train_X.shape[0]  # Number of examples
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        train_X_shuffled = train_X[shuffled_indices]
        train_Y_shuffled = train_Y[shuffled_indices]

        for i in range(0, m, batch_size):
            X_batch = train_X_shuffled[i:i + batch_size]
            Y_batch = train_Y_shuffled[i:i + batch_size]

            # Forward propagation
            Z = np.dot(X_batch, W) + b
            Y_pred = softmax(Z)

            loss = cross_entropy_loss(Y_pred, Y_batch)

            # Backward propagation (Gradient Descent)
            m_batch = X_batch.shape[0]
            dZ = Y_pred - Y_batch
            dW = np.dot(X_batch.T, dZ) / m_batch
            db = np.sum(dZ, axis=0, keepdims=True) / m_batch

            W -= learning_rate * dW
            b -= learning_rate * db

        Z_train = np.dot(train_X, W) + b
        Y_train_pred = softmax(Z_train)
        train_acc = accuracy(Y_train_pred, train_Y)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Training Accuracy: {train_acc * 100:.2f}%')

    return W, b


def test_perceptron(test_X, test_Y, W, b):
    Z_test = np.dot(test_X, W) + b
    Y_test_pred = softmax(Z_test)
    test_acc = accuracy(Y_test_pred, test_Y)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')


print("Initial Accuracy (Before Training):")
test_perceptron(test_X, test_Y, W, b)

W, b = train_perceptron(train_X, train_Y, W, b, epochs=200, batch_size=100)

print("Final Accuracy (After Training):")
test_perceptron(test_X, test_Y, W, b)
