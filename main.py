"""Pylight recognize handwritten numbers using rays of light and the KNN algorithm."""

from typing import Any, Generator, Union

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
        variation_shape_top_abs, variation_shape_top_rel, diff_start_end_top = (
            self.get_variation_shape(image, "top")
        )
        variation_shape_bottom_abs, variation_shape_bottom_rel, diff_start_end_bottom = (
            self.get_variation_shape(image, "bottom")
        )
        variation_shape_left_abs, variation_shape_left_rel, diff_start_end_left = (
            self.get_variation_shape(image, "left")
        )
        variation_shape_right_abs, variation_shape_right_rel, diff_start_end_right = (
            self.get_variation_shape(image, "right")
        )
        coordinates = np.array(
            [
                variation_shape_top_abs,
                variation_shape_top_rel,
                diff_start_end_top,
                variation_shape_bottom_abs,
                variation_shape_bottom_rel,
                diff_start_end_bottom,
                variation_shape_left_abs,
                variation_shape_left_rel,
                diff_start_end_left,
                variation_shape_right_abs,
                variation_shape_right_rel,
                diff_start_end_right,
            ]
        )
        return coordinates

    def get_variation_shape(
        self, image: npt.NDArray[np.float64], side: str
    ) -> tuple[np.float64, np.float64, np.float64]:
        """Get the variation of the shape of the number from one side.

        Args:
            image (npt.NDArray[np.float64]): The image.
            side (str): The side of the number ("left", "right", "top", "bottom").

        Returns:
            np.float64: A value representing the variation of the shape of the number.
        """
        if side not in {"left", "right", "top", "bottom"}:
            raise ValueError("Invalid side. Must be 'left', 'right', 'top', or 'bottom'.")

        variation_abs = 0
        variation_rel = 0
        start = 0
        end = 0
        previous_position = None

        if side in {"left", "right"}:
            axis = 1
            reverse = side == "right"
        else:
            axis = 0
            reverse = side == "bottom"

        for line in image.T if axis == 0 else image:
            positions = np.where(line > 0)[0]
            if reverse:
                positions = positions[::-1]

            if len(positions) > 0:
                current_position = positions[0]
                if previous_position is not None:
                    variation_abs += abs(current_position - previous_position)
                    variation_rel += current_position - previous_position
                if start is None:
                    start = current_position
                previous_position = current_position

                end = current_position

        return np.float64(variation_abs), np.float64(variation_rel), np.float64(end - start)

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
