"""SVHN data loading."""

import numpy as np


def load_svhn(
    root: str = "data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load channel-normalized torchvision SVHN arrays."""
    from torchvision.datasets import SVHN

    train = SVHN(root, split="train", download=True)
    test = SVHN(root, split="test", download=True)
    mean = np.array((.4377, .4438, .4728), dtype=np.float32)[None, :, None, None]
    std = np.array((.1980, .2010, .1970), dtype=np.float32)[None, :, None, None]

    def arrays(dataset):
        X = dataset.data.astype(np.float32) / 255
        return (X - mean) / std, np.asarray(dataset.labels, dtype=np.int64)

    X_train, y_train = arrays(train)
    X_test, y_test = arrays(test)
    return X_train, y_train, X_test, y_test
