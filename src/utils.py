"""Utility functions for image processing and visualization"""

import torch
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


def load_image_from_file(file_path, target_size=(384, 384)):
    """
    Load and resize image from local file

    Args:
        file_path: Path to the image file
        target_size: Target size for resizing (default: 384x384)

    Returns:
        PIL.Image: Resized image in RGB format
    """
    img = Image.open(file_path).convert("RGB")
    resized_image = img.resize(target_size)
    return resized_image


def replace_with_padding(input_tensor, target_values, padding_token_id):
    """
    Replace target values with padding token

    Args:
        input_tensor: Input tensor to modify
        target_values: Values to replace
        padding_token_id: Padding token ID

    Returns:
        torch.Tensor: Modified tensor
    """
    mask = torch.isin(input_tensor, target_values)
    input_tensor[mask] = padding_token_id
    return input_tensor


def setup_colormap():
    """
    Create custom colormap for visualization

    Returns:
        LinearSegmentedColormap: Custom blue gradient colormap
    """
    return LinearSegmentedColormap.from_list(
        'custom blue',
        [(0, '#ffffff'), (0.25, '#252b36'), (1, '#000000')],
        N=256
    )
