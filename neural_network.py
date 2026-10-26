"""A fully-connected neural network built with NumPy only.

Architecture (default): 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
Loss: cross-entropy.  Optimizer: mini-batch gradient descent.
"""
import numpy as np


# ---------- activation functions ----------
def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


class NeuralNetwork:
    def __init__(self, layer_sizes=(784, 128, 64, 10), seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(layer_sizes)
        # He initialization works well with ReLU
        self.W = [rng.standard_normal((i, o)) * np.sqrt(2.0 / i)
                  for i, o in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros((1, o)) for o in self.sizes[1:]]

    # ---------- forward propagation ----------
    def forward(self, X):
        """Return class probabilities; caches values needed for backprop."""
        self.Z, self.A = [], [X]
        a = X
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            self.Z.append(z)
            a = softmax(z) if l == len(self.W) - 1 else relu(z)
            self.A.append(a)
        return a

    # ---------- loss ----------
    @staticmethod
    def loss(probs, y_onehot):
        """Mean cross-entropy loss."""
        return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))

    # ---------- backpropagation ----------
    def backward(self, y_onehot):
        """Return gradients (dW, db) for every layer."""
        m = y_onehot.shape[0]
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        delta = (self.A[-1] - y_onehot) / m  # softmax + cross-entropy shortcut
        for l in reversed(range(len(self.W))):
            dW[l] = self.A[l].T @ delta
            db[l] = delta.sum(axis=0, keepdims=True)
            if l > 0:
                delta = (delta @ self.W[l].T) * relu_grad(self.Z[l - 1])
        return dW, db

    def _update(self, dW, db, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    # ---------- prediction / metrics ----------
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # ---------- training loop ----------
    def fit(self, X, y, X_val, y_val, epochs=10, lr=0.1, batch_size=64, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        n_classes = self.sizes[-1]
        Y = one_hot(y, n_classes)
        Yv = one_hot(y_val, n_classes)
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            idx = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                b = idx[start:start + batch_size]
                self.forward(X[b])
                dW, db = self.backward(Y[b])
                self._update(dW, db, lr)

            history["train_loss"].append(self.loss(self.forward(X), Y))
            history["val_loss"].append(self.loss(self.forward(X_val), Yv))
            history["train_acc"].append(self.accuracy(X, y))
            history["val_acc"].append(self.accuracy(X_val, y_val))
            if verbose:
                print(f"Epoch {epoch:2d}/{epochs} | "
                      f"loss {history['train_loss'][-1]:.4f} | val loss {history['val_loss'][-1]:.4f} | "
                      f"acc {history['train_acc'][-1]:.4f} | val acc {history['val_acc'][-1]:.4f}")
        return history

    # ---------- save / load ----------
    def save(self, path):
        np.savez(path, sizes=self.sizes, **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(tuple(d["sizes"]))
        net.W = [d[f"W{i}"] for i in range(len(net.W))]
        net.b = [d[f"b{i}"] for i in range(len(net.b))]
        return net
