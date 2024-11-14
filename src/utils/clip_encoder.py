from typing import Union, List
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP encoder with specified model.

        Args:
            model_name (str): Name of the CLIP model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _load_image_from_url(url: str) -> Image.Image:
        """
        Load an image from a URL.

        Args:
            url (str): URL of the image

        Returns:
            PIL.Image: Loaded image

        Raises:
            ValueError: If image cannot be loaded from URL
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {str(e)}")

    @staticmethod
    def _load_image_from_path(path: Union[str, Image.Image]) -> Image.Image:
        if isinstance(path, str):
            img = Image.open(path).convert('RGB')
        elif isinstance(path, Image.Image):
            img = path
        else:
            raise ValueError(f"Expected path or PIL.Image as the input but get {str(path)}")
        return img

    def encode_text(self, text: str) -> List[float]:
        """
        Encode text using CLIP model.

        Args:
            text (str): Input text to encode

        Returns:
            np.ndarray: Text embedding
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features[0].cpu().tolist()

    def encode_image(self, images: List[Union[str, Image.Image]], is_url: bool = True) -> List[float]:
        """
        Encode image using CLIP model.

        Args:
            images (Union[str, Image.Image]): Input images as a list of URLs or PIL Images
            is_url (bool): Whether the image input is a URL

        Returns:
            np.ndarray: Image embedding

        Raises:
            ValueError: If image cannot be processed
        """
        try:
            # Handle image input
            if is_url:
                for image in images:
                    if not isinstance(image, str):
                        raise ValueError("URL must be a string")
                input_images = list(map(self._load_image_from_url, images))
            else:
                input_images = list(map(self._load_image_from_path, images))

            # Process image
            inputs = self.processor(images=input_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # because we use `dot products`/`cosine` at the end
            # it would make sense to use mean of the vectors as the representor
            return image_features.mean(axis=0).cpu().tolist()

        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
