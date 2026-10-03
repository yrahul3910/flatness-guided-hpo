"""Regression checks for image labels, normalization, and channel layout."""

import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override
from unittest.mock import patch

import numpy as np
from keras import backend
from scipy.io import savemat

from image.src.config import Config
from image.src.util import get_data, get_model


class ImageDataTests(unittest.TestCase):
    """Exercise preprocessing using small image fixtures and real SVHN MAT files."""

    @override
    def setUp(self) -> None:
        """Restore the global Keras image layout after each test."""
        self.addCleanup(backend.set_image_data_format, backend.image_data_format())
        backend.set_image_data_format("channels_last")

    def test_svhn_zero_label_has_ten_classes(self) -> None:
        """Map the raw digit-zero label in both splits before one-hot encoding."""
        images = np.zeros((32, 32, 3, 2), dtype=np.uint8)
        train_labels = np.array([[10], [9]], dtype=np.uint8)
        test_labels = np.array([[9], [10]], dtype=np.uint8)

        with TemporaryDirectory() as directory, chdir(directory):
            savemat(Path("train_32x32.mat"), {"X": images, "y": train_labels})
            savemat(Path("test_32x32.mat"), {"X": images, "y": test_labels})
            data = get_data("svhn")

        np.testing.assert_equal(data.y_train.shape, (2, 10))
        np.testing.assert_equal(data.y_test.shape, (2, 10))
        np.testing.assert_array_equal(data.y_train.argmax(axis=1), [0, 9])
        np.testing.assert_array_equal(data.y_test.argmax(axis=1), [9, 0])

    def test_rgb_channel_layout_preserves_pixels(self) -> None:
        """Keep each RGB pixel in its channel and accept valid image geometry.

        Distinct values across spatial positions and channels expose a reshape that scrambles RGB pixels.
        The two-block configuration must be accepted with either supported channel layout.
        """
        images = np.arange(2 * 32 * 32 * 3, dtype=np.uint16).reshape(2, 32, 32, 3).astype(np.uint8)
        labels = np.array([[0], [9]], dtype=np.uint8)
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
        expected = (images.astype(np.float32) / 255 - mean) / std
        config = Config(
            n_filters=2, kernel_size=3, padding="same", n_blocks=2, n_units=4, learning_rate=0.001, weight_decay=0.0001
        )

        for data_format in ("channels_last", "channels_first"):
            with self.subTest(data_format=data_format):
                backend.set_image_data_format(data_format)

                with patch("image.src.util.get_cifar10", return_value=((images, labels), (images, labels))):
                    data = get_data("cifar10")

                expected_layout = expected if data_format == "channels_last" else expected.transpose(0, 3, 1, 2)
                np.testing.assert_allclose(data.x_train, expected_layout, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(data.x_test, expected_layout, rtol=1e-6, atol=1e-6)
                np.testing.assert_equal(data.x_train.dtype, np.dtype(np.float32))
                np.testing.assert_equal(data.x_test.dtype, np.dtype(np.float32))
                np.testing.assert_equal(get_model(data, config, 10, "cifar10") is None, desired=False)

    def test_mnist_grayscale_layout_and_normalization(self) -> None:
        """Normalize grayscale pixels without promoting the image arrays to float64."""
        images = np.arange(2 * 28 * 28, dtype=np.uint16).reshape(2, 28, 28).astype(np.uint8)
        labels = np.array([0, 9], dtype=np.uint8)
        expected = (images.astype(np.float32) / 255 - np.float32(0.1307)) / np.float32(0.3081)

        for data_format in ("channels_last", "channels_first"):
            with self.subTest(data_format=data_format):
                backend.set_image_data_format(data_format)

                with patch("image.src.util.get_mnist", return_value=((images, labels), (images, labels))):
                    data = get_data("mnist")

                expected_layout = (
                    expected[..., np.newaxis] if data_format == "channels_last" else expected[:, np.newaxis]
                )
                np.testing.assert_allclose(data.x_train, expected_layout, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(data.x_test, expected_layout, rtol=1e-6, atol=1e-6)
                np.testing.assert_equal(data.x_train.dtype, np.dtype(np.float32))
                np.testing.assert_equal(data.x_test.dtype, np.dtype(np.float32))
