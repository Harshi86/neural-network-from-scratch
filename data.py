"""MNIST loader that uses only the standard library + NumPy.

Downloads the 4 official IDX files on first run and caches them in ./data
"""
import gzip
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).with_name("data")
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename):
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / filename
    if target.exists():
        return target
    for base in MIRRORS:
        try:
            print(f"Downloading {base + filename} ...")
            urllib.request.urlretrieve(base + filename, target)
            return target
        except Exception as e:  # try the next mirror
            print(f"  failed ({e})")
    raise RuntimeError(
        f"Could not download {filename}. Download the 4 MNIST .gz files manually "
        f"and place them in {DATA_DIR}"
    )


def _read_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float64) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)


def load_mnist():
    """Return (X_train, y_train, X_test, y_test); pixels scaled to [0, 1]."""
    X_train = _read_images(_download(FILES["train_images"]))
    y_train = _read_labels(_download(FILES["train_labels"]))
    X_test = _read_images(_download(FILES["test_images"]))
    y_test = _read_labels(_download(FILES["test_labels"]))
    return X_train, y_train, X_test, y_test
