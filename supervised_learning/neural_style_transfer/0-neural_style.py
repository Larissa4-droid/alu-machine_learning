#!/usr/bin/env python3
"""Module that defines the NST class for Neural Style Transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Class that performs tasks for neural style transfer"""
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST

        Args:
            style_image (np.ndarray): The style reference image (h, w, 3)
            content_image (np.ndarray): The content reference image (h, w, 3)
            alpha (float/int): Weight for content cost (default 1e4)
            beta (float/int): Weight for style cost (default 1)
        """

        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so its pixel values are in [0, 1]
        and its largest side is 512 pixels.

        Args:
            image (np.ndarray): Image array of shape (h, w, 3)

        Returns:
            tf.Tensor: Scaled image tensor of shape (1, h_new, w_new, 3)
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        max_dim = 512
        scale = max_dim / max(h, w)
        h_new = int(round(h * scale))
        w_new = int(round(w * scale))

        image_expanded = tf.expand_dims(image, axis=0)

        resized_image = tf.image.resize_bicubic(
            image_expanded,
            size=[h_new, w_new]
        )

        scaled_image = resized_image / 255.0
        scaled_image = tf.clip_by_value(scaled_image, 0.0, 1.0)

        return scaled_image
