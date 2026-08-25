"""CIFAR-10 data loading."""

import numpy as np


from topotrace.mnist import DATA_ROOT


def load_cifar10(
    root: str = DATA_ROOT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load channel-normalized torchvision CIFAR-10 arrays."""
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root, train=True, download=True)
    test = CIFAR10(root, train=False, download=True)
    mean = np.array((.4914, .4822, .4465), dtype=np.float32)[None, :, None, None]
    std = np.array((.2470, .2435, .2616), dtype=np.float32)[None, :, None, None]

    def arrays(dataset):
        X = dataset.data.transpose(0, 3, 1, 2).astype(np.float32) / 255
        return (X - mean) / std, np.asarray(dataset.targets, dtype=np.int64)

    X_train, y_train = arrays(train)
    X_test, y_test = arrays(test)
    return X_train, y_train, X_test, y_test
