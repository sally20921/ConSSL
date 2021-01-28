"""
Collection of PyTorchLightning models
"""

from CSSL.models.autoencoders.basic_ae.basic_ae_module import AE  # noqa: F401
from CSSL.models.autoencoders.basic_vae.basic_vae_module import VAE  # noqa: F401
from CSSL.models.mnist_module import LitMNIST  # noqa: F401
from CSSL.models.regression import LinearRegression, LogisticRegression  # noqa: F401
from CSSL.models.vision import PixelCNN, SemSegment, UNet  # noqa: F401
from CSSL.models.vision.image_gpt.igpt_module import GPT2, ImageGPT  # noqa: F401

__all__ = [
    "AE",
    "VAE",
    "LitMNIST",
    "LinearRegression",
    "LogisticRegression",
    "PixelCNN",
    "SemSegment",
    "UNet",
    "GPT2",
    "ImageGPT",
]
