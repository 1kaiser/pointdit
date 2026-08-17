# coding=utf-8

import torch
from PIL import Image, ImageDraw
import numpy as np


def add_text_border(image_tensor, text, border_height=20):
    """
    Args:
        image_tensor: (C, H, W) tensor, normalized 0-1
        text: string to draw
        border_height: height of the text area in pixels
    """
    # 1. Convert Tensor to PIL
    img_np = image_tensor.cpu().detach().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    old_img = Image.fromarray(img_np)
    w, h = old_img.size

    # 2. Create Canvas
    new_img = Image.new(old_img.mode, (w, h + border_height), (255, 255, 255))
    new_img.paste(old_img, (0, border_height))
    
    # 3. Draw Text Centered
    draw = ImageDraw.Draw(new_img)
    
    # Define the center point of the border area
    center_x = w / 2
    center_y = border_height / 2
    
    # anchor="mm" stands for "Middle-Middle" (horizontal & vertical centering)
    # This centers the text exactly at the coordinates provided
    draw.text((center_x, center_y), text, fill=(0, 0, 0), anchor="mm")
    
    # 4. Return Tensor
    result_np = np.array(new_img)
    return torch.from_numpy(result_np).permute(2, 0, 1).float() / 255.0
