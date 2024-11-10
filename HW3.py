import numpy as np
from torchvision.datasets import MNIST

def load_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=False,
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
input_size = 784
hidden_size = 100
output_size = 10
learning_rate = 0.1
dropout_rate = 0.2  # Dropout rate for hidden layer

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m


def accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)


def forward_propagation(X, W1, b1, W2, b2, dropout_mask=None):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    if dropout_mask is not None:
        A1 *= dropout_mask  # Apply dropout mask
        A1 /= (1 - dropout_rate)  # Scale to maintain expected value

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backpropagation(X, Y, Z1, A1, A2, W1, W2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dA1 *= relu_derivative(Z1)

    dW1 = np.dot(X.T, dA1) / m
    db1 = np.sum(dA1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


def train_mlp(train_X, train_Y, W1, b1, W2, b2, epochs=100, batch_size=100):
    m = train_X.shape[0]
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        train_X_shuffled = train_X[shuffled_indices]
        train_Y_shuffled = train_Y[shuffled_indices]

        for i in range(0, m, batch_size):
            X_batch = train_X_shuffled[i:i + batch_size]
            Y_batch = train_Y_shuffled[i:i + batch_size]

            # Apply dropout to the hidden layer
            dropout_mask = (np.random.rand(X_batch.shape[0], hidden_size) > dropout_rate).astype(float)

            Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2, dropout_mask)

            # Compute gradients
            dW1, db1, dW2, db2 = backpropagation(X_batch, Y_batch, Z1, A1, A2, W1, W2)

            # Update weights and biases
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        _, _, _, train_A2 = forward_propagation(train_X, W1, b1, W2, b2)
        train_acc = accuracy(train_A2, train_Y)
        print(f'Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_acc * 100:.2f}%')

    return W1, b1, W2, b2


def test_mlp(test_X, test_Y, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(test_X, W1, b1, W2, b2)
    test_acc = accuracy(A2, test_Y)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')


print("Initial Accuracy (Before Training):")
test_mlp(test_X, test_Y, W1, b1, W2, b2)

W1, b1, W2, b2 = train_mlp(train_X, train_Y, W1, b1, W2, b2, epochs=100, batch_size=100)

print("Final Accuracy (After Training):")
test_mlp(test_X, test_Y, W1, b1, W2, b2)
