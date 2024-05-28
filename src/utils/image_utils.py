from typing import List

from PIL import Image


def load_image_from_path(img_path: str) -> Image.Image:
    """Loads an image from the specified file path and returns a PIL image."""
    return Image.open(img_path).convert("RGB")


def load_images_from_paths(img_paths: List[str]) -> List[Image.Image]:
    """Loads a list of images from the given file paths and returns a list of PIL images."""
    return [load_image_from_path(img_path) for img_path in img_paths]
