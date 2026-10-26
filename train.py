"""Train the NumPy neural network on MNIST and visualize the results."""
import argparse

import matplotlib
import numpy as np

from data import load_mnist
from neural_network import NeuralNetwork


def confusion_matrix(y_true, y_pred, n=10):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def plot_results(history, net, X_test, y_test, path):
    import matplotlib.pyplot as plt

    preds = net.predict(X_test)
    fig = plt.figure(figsize=(14, 9))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    ax.set(title="Loss (cross-entropy)", xlabel="Epoch", ylabel="Loss")
    ax.legend()

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(epochs, history["train_acc"], label="train")
    ax.plot(epochs, history["val_acc"], label="validation")
    ax.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax.legend()

    ax = fig.add_subplot(2, 3, 3)
    cm = confusion_matrix(y_test, preds)
    ax.imshow(cm, cmap="Blues")
    ax.set(title="Confusion matrix (test)", xlabel="Predicted", ylabel="True",
           xticks=range(10), yticks=range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    rng = np.random.default_rng(1)
    for k, i in enumerate(rng.choice(len(X_test), 3, replace=False)):
        ax = fig.add_subplot(2, 6, 7 + k * 2)
        ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
        ok = preds[i] == y_test[i]
        ax.set_title(f"pred {preds[i]} (true {y_test[i]})", color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    wrong = np.where(preds != y_test)[0]
    for k, i in enumerate(wrong[:3]):
        ax = fig.add_subplot(2, 6, 8 + k * 2)
        ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"MISS: pred {preds[i]} (true {y_test[i]})", color="red", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"NumPy Neural Network - Test accuracy: {np.mean(preds == y_test):.2%}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plots to {path}")
    return plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    args = p.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    X_train, y_train, X_test, y_test = load_mnist()
    # hold out the last 10,000 training images for validation
    X_val, y_val = X_train[-10000:], y_train[-10000:]
    X_train, y_train = X_train[:-10000], y_train[:-10000]
    print(f"Train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")

    net = NeuralNetwork((784, 128, 64, 10))
    history = net.fit(X_train, y_train, X_val, y_val,
                      epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    print(f"\nFinal test accuracy: {net.accuracy(X_test, y_test):.4f}")
    net.save("model.npz")
    plt = plot_results(history, net, X_test, y_test, "results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
