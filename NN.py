import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


MODEL_FILE = "mnist_model.npz"


# Load the MNIST dataset
def load_data():
    print("Loading MNIST dataset...")

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False
    )

    X = mnist.data
    y = mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Normalize pixels from 0-255 to 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test, y_train, y_test


def one_hot(y, num_classes=10):
    """
    5 ---> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] (this is an example for label 5)
    """
    return np.eye(num_classes)[y]


def initialize_parameters():
    np.random.seed(42)
    W1 = np.random.randn(128, 784) * 0.01
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(10, 128) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return Z_exp / np.sum(Z_exp, axis=0, keepdims=True)


def forward_pass(X, W1, b1, W2, b2):
    # Hidden layer
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)

    # Output layer computation
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def compute_loss(A2, Y):
    m = Y.shape[0]
    return -np.sum(Y * np.log(A2.T + 1e-8)) / m


def backward_pass(X, Y, Z1, A1, A2, W2):
    m = X.shape[0]

    # Gradient at the output layer
    dZ2 = A2 - Y.T  # Difference between prediction and true labels
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Gradient at the hidden layer
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


def train(X_train, y_train_onehot, epochs=20, alpha=0.01, batch_size=64):
    W1, b1, W2, b2 = initialize_parameters()

    for epoch in range(epochs):
        # Shuffle the training data
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        y_shuffled = y_train_onehot[permutation]

        # Process data in mini-batches
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Forward propagation
            Z1, A1, Z2, A2 = forward_pass(X_batch, W1, b1, W2, b2)

            # Compute the loss
            loss = compute_loss(A2, y_batch)

            # Backpropagation
            dW1, db1, dW2, db2 = backward_pass(X_batch, y_batch, Z1, A1, A2, W2)

            # Update parameters using gradient descent
            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Display the loss at the end of each epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    return W1, b1, W2, b2


def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=0)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def save_model(W1, b1, W2, b2):
    np.savez(
        MODEL_FILE,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2
    )
    print(f"Model saved to {MODEL_FILE}")


def load_model():
    model = np.load(MODEL_FILE)

    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]

    return W1, b1, W2, b2


if __name__ == "__main__":

    if os.path.exists(MODEL_FILE):
        print("Saved model found.")
        print("Loading model...")

        W1, b1, W2, b2 = load_model()

    else:
        print("No saved model found.")
        print("Training neural network...")

        X_train, X_test, y_train, y_test = load_data()

        y_train_onehot = one_hot(y_train)

        W1, b1, W2, b2 = train(X_train, y_train_onehot)

        save_model(W1, b1, W2, b2)

        # Test accuracy
        y_pred_test = predict(X_test, W1, b1, W2, b2)
        test_acc = accuracy(y_test, y_pred_test)

        print(f"Test Accuracy: {test_acc * 100:.2f}%")

    print("Ready!")
