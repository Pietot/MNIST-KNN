"""Pylight recognize handwritten numbers using rays of light and the KNN algorithm."""

from typing import Any, Generator, Union

import pickle
import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from numpy import typing as npt


class KDT:
    """Class representing a k-dimensional tree."""

    def __init__(self, k_nearest: int = 5) -> None:
        self.coordinates: list[npt.NDArray[np.float64]] = []
        self.images, self.labels = load_train_mnist()
        self.k_nearest = k_nearest
        self.build_kdtree()

    def __iter__(self) -> Generator[npt.NDArray[np.float64], None, None]:
        yield from self.images

    def __str__(self) -> str:
        return f"KDT with {len(self.coordinates)} points"

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

    def build_kdtree(self) -> None:
        """Build a KDTree from the coordinates calculated for each image."""
        for image in self:
            coordinates = self.get_coordinates(image)
            self.add(coordinates)

        points = np.array(self.coordinates)
        self.kdtree = cKDTree(points)

    def get_coordinates(self, image: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gets the 9 coordinates for all images.

        Args:
            image (npt.NDArray[np.float64]): The image.
        """
        ray_vertical = self.raytracing_vertical(image)
        ray_horizontal = self.raytracing_horizontal(image)
        light_from_top = self.light_from(image, "top")
        light_from_bottom = self.light_from(image, "bottom")
        light_from_left = self.light_from(image, "left")
        light_from_right = self.light_from(image, "right")
        size_number = self.get_size_number(image)
        coordinates = np.array(
            [
                ray_vertical,
                ray_horizontal,
                light_from_top,
                light_from_bottom,
                light_from_left,
                light_from_right,
                size_number,
            ]
        )
        return coordinates

    def raytracing_vertical(self, image: npt.NDArray[np.float64]) -> np.float64:
        """Parse each column of the image and see if all value are 0.

        Args:
            image (npt.NDArray[np.float64]): The image to parse.

        Returns:
            np.float64: The percentage of rays that reach the bottom.
        """
        reaches = np.sum(np.all(image == 0, axis=0))
        return np.float64(reaches / image.shape[1])

    def raytracing_horizontal(self, image: npt.NDArray[np.float64]) -> np.float64:
        """Parse each line of the image and see if all values are 0.

        Args:
            image (npt.NDArray[np.float64]): The image to parse.

        Returns:
            np.float64: The percentage of rays that reach the bottom.
        """
        reaches = np.sum(np.all(image == 0, axis=1))
        return np.float64(reaches / image.shape[1])

    def light_from(self, image: npt.NDArray[np.float64], source: str) -> np.float64:
        """Parse each column of the image, throw a ray from the source and see how many pixels
        are crossed by the ray.

        Args:
            image (npt.NDArray[np.float64]): The image to parse.

        Returns:
            np.float64: The percentage of pixels crossed by the ray.
        """
        axis = 0 if source in {"top", "bottom"} else 1
        if source in {"bottom", "right"}:
            image = np.flip(image, axis=axis)

        zero_pixels = np.sum(image == 0)
        hit_number = np.cumsum(image != 0, axis=axis) > 0
        pixels_crossed = np.sum((image == 0) & ~hit_number)

        return np.float64(pixels_crossed / zero_pixels)

    def get_size_number(self, image: npt.NDArray[np.float64]) -> np.float64:
        """Get the number of pixels that belong to the number.

        Args:
            image (npt.NDArray[np.float64]): The image.

        Returns:
            np.float64: The percentage of pixels that belong to the number.
        """
        return np.float64(np.sum(image > 0) / image.size)

    def add(self, coordinates: npt.NDArray[np.float64]) -> None:
        """Add a point into the KDT.

        Args:
            coordinates (npt.NDArray[np.float64]): The 9 dimensions coordinates of the point.
        """
        self.coordinates.append(coordinates)

    def predict(self, image: npt.NDArray[np.float64]) -> int:
        """Predict the value of the image using the KNN algorithm.

        Args:
            image (npt.NDArray[np.float64]): The image to predict.

        Returns:
            int: The predicted value.
        """
        coordinates = self.get_coordinates(image)
        _, indices = self.kdtree.query(coordinates, k=self.k_nearest)
        nearest_values = self.labels[indices]

        return int(np.bincount(nearest_values).argmax())


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


def load_test_mnist() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Load the test MNIST dataset.

    Returns:
        tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
            The images and the labels.
    """
    with open(r"assets\t10k-images-idx3-ubyte", "rb") as f:
        f.read(4)  # Ignore magic field
        num_images = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")

        image_data = f.read(num_images * num_rows * num_cols)
        images = np.frombuffer(image_data, dtype=np.uint8).reshape((num_images, num_rows, num_cols))

    with open(r"assets\t10k-labels-idx1-ubyte", "rb") as f:
        f.read(4)  # Ignore magic field
        num_labels = int.from_bytes(f.read(4), "big")
        assert num_labels == num_images, "Number of labels does not match number of images"

        label_data = f.read(num_labels)
        labels = np.frombuffer(label_data, dtype=np.uint8)

    return images, labels


if __name__ == "__main__":
    tree = KDT()
    test_images, test_labels = load_test_mnist()
    correct: int = 0
    for img, lab in zip(test_images, test_labels):
        prediction: int = tree.predict(img)
        if prediction == lab:
            correct += 1
    print(f"Accuracy: {correct / len(test_images) * 100:.2f}%")
