import cv2
import torch
import numpy as np

def img_crop(img):
    return img[30:-12, :, :]

def downsample(img):
    return img[::2, ::2]

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def normalize_grayscale(img):
    return (img - 128) / 128 - 1  

def preprocess_state(state, device, image_shape=(80, 80)):
    if isinstance(state, tuple):
        state = state[0]
    if not isinstance(state, np.ndarray):
        raise ValueError("State must be an ndarray")
    
    state = img_crop(state)
    state = downsample(state)
    state = to_grayscale(state)
    state = normalize_grayscale(state)
    
    state = cv2.resize(state, image_shape)  # Resize to the desired shape (80, 80)
    state = np.expand_dims(state, axis=0)  # Add channel dimension
    state = np.expand_dims(state, axis=0)  # Add batch dimension
    
    return torch.tensor(state, dtype=torch.float32).to(device)
