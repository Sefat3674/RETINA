import cv2
import numpy as np

def crop_circle(img):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    radius = min(center[0], center[1])
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist <= radius
    if img.ndim == 3:
        mask = np.stack([mask]*3, axis=-1)
    img[~mask] = 0
    return img

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def sharpen_image(img, sigma=10):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)

def resize_normalize(img, size=(224, 224)):
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_circle(img)
    img = apply_clahe(img)
    img = sharpen_image(img)
    processed = resize_normalize(img)
    return np.expand_dims(processed, axis=0), img


