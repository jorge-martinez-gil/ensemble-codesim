# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

from PIL import Image, ImageDraw, ImageFont
import imagehash

def code_to_image(code, image_size=(300, 100)):
    try:
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()  # You can adjust the font
        draw.text((10, 10), code, fill='black', font=font)
        return img
    except Exception:
        return None  # Return None if the image creation fails

def similarity(code1, code2):
    image1 = code_to_image(code1)
    image2 = code_to_image(code2)
    
    if image1 is None or image2 is None:
        return 0  # Return 0 if image creation failed

    try:
        # Compute perceptual hash values for the images
        hash1 = imagehash.phash(image1)
        hash2 = imagehash.phash(image2)
        # Compute hamming distance between hash values
        hamming_distance = hash1 - hash2  # Lower values indicate higher similarity
        similarity_ratio = 1 - (hamming_distance / len(hash1.hash) ** 2)
        return similarity_ratio
    except Exception:
        return 0  # Return 0 if hashing or comparison fails
