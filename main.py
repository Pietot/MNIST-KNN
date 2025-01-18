"""Pylight recognize handwritten numbers using rays of light and the KNN algorithm."""

from typing import Any, Generator, Union

import numpy as np

from numpy import typing as npt


class Field:
    """Class representing a field of points with associated images and labels."""

    def __init__(self) -> None:
        self.points: list[tuple[npt.NDArray[np.float64], int]] = []
        self.images, self.labels = load_train_mnist()

    def __iter__(self) -> Generator[tuple[npt.NDArray[np.float64], int], None, None]:
        for image, label in zip(self.images, self.labels):
            yield image, label

    def __str__(self) -> str:
        return f"Field with {len(self.points)} points"

    def __getitem__(
        self, index: Union[int, slice, Any]
    ) -> Union[tuple[npt.NDArray[np.float64], int], list[tuple[npt.NDArray[np.float64], int]]]:
        if isinstance(index, int):
            return self.images[index], self.labels[index]
        if isinstance(index, slice):
            return [
                (self.images[i], self.labels[i]) for i in range(*index.indices(len(self.images)))
            ]
        raise TypeError("Index must be an integer or a slice")

    def add(self, coordinates: npt.NDArray[np.float64], value: int) -> None:
        """Add a point into the 9 dimension field.

        Args:
            coordinates (npt.NDArray[np.float64]): The 9 dimensions coordinates of the point.
            value (int): The value of the point (0-9)
        """
        self.points.append((coordinates, value))


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


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    field = Field()
    print(field[2])
