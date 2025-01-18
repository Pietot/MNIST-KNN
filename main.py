"""Pylight recognize handwritten numbers using rays of light and the KNN algorithm."""

import numpy as np

from numpy import typing as npt


def load_train_mnist() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Load the training MNIST dataset.

    Returns:
        tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
            The images and the labels.
    """
    with open(r"assets\train-images-idx3-ubyte", "rb") as f:
        f.read(4)  # Ignore magic field
        num_images = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")

        image_data = f.read(num_images * num_rows * num_cols)
        images = np.frombuffer(image_data, dtype=np.uint8).reshape((num_images, num_rows, num_cols))

    with open(r"assets\train-labels-idx1-ubyte", "rb") as f:
        f.read(4)  # Ignore magic field
        num_labels = int.from_bytes(f.read(4), "big")
        assert num_labels == num_images, "Number of labels does not match number of images"

        label_data = f.read(num_labels)
        labels = np.frombuffer(label_data, dtype=np.uint8)

    return images, labels