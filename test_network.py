import numpy as np

from neural_network import NeuralNetwork, one_hot, softmax


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.randn(5, 10) * 50)
    assert np.allclose(p.sum(axis=1), 1)


def test_forward_shape():
    net = NeuralNetwork((20, 8, 4))
    assert net.forward(np.random.randn(7, 20)).shape == (7, 4)


def test_gradient_check():
    """Backprop gradients must match numerical gradients."""
    rng = np.random.default_rng(0)
    net = NeuralNetwork((6, 5, 3), seed=1)
    X = rng.standard_normal((4, 6))
    Y = one_hot(np.array([0, 1, 2, 1]), 3)
    net.forward(X)
    dW, _ = net.backward(Y)
    eps = 1e-6
    for l in range(len(net.W)):
        for idx in [(0, 0), (1, 1), (2, 0)]:
            old = net.W[l][idx]
            net.W[l][idx] = old + eps
            up = net.loss(net.forward(X), Y)
            net.W[l][idx] = old - eps
            down = net.loss(net.forward(X), Y)
            net.W[l][idx] = old
            assert abs((up - down) / (2 * eps) - dW[l][idx]) < 1e-6


def test_learns_simple_problem():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((3, 10)) * 3
    y = rng.integers(0, 3, 600)
    X = centers[y] + rng.standard_normal((600, 10))
    net = NeuralNetwork((10, 16, 3))
    net.fit(X[:500], y[:500], X[500:], y[500:], epochs=30, lr=0.1, verbose=False)
    assert net.accuracy(X[500:], y[500:]) > 0.95
