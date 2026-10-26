# Basic Neural Network from Scratch (NumPy only)

A neural network that classifies handwritten digits (MNIST), built **using only NumPy** - no TensorFlow, PyTorch or scikit-learn. Matplotlib is used just for visualization.

## What's implemented by hand
- **Forward propagation** with ReLU hidden layers and a Softmax output
- **Cross-entropy loss** and **accuracy**
- **Backpropagation** (verified against numerical gradients in the tests)
- **Mini-batch gradient descent** and He weight initialization
- **Visualizations:** loss curve, accuracy curve, confusion matrix, sample predictions and misclassified digits

## Architecture
`784 (28x28 pixels) -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)`

## Project structure
| File | Purpose |
|------|---------|
| `neural_network.py` | The network: layers, activations, forward/backward pass, training loop |
| `data.py` | Downloads and parses MNIST using only NumPy + stdlib |
| `train.py` | Trains the model and generates `results.png` |
| `test_network.py` | Unit tests, including a gradient check |

## Run
```bash
pip install -r requirements.txt
python train.py                 # trains, saves model.npz and results.png
python train.py --epochs 20 --lr 0.05 --batch-size 128
python train.py --no-show       # don't open a plot window
pytest                          # run tests
```
The first run downloads MNIST (~11 MB) into `data/`. Expected test accuracy is around 97-98%.

## Results
Add your `results.png` here after training:

`![results](results.png)`

**Tech stack:** Python, NumPy, Matplotlib
