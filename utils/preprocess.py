import cv2
import numpy as np

def preprocess_image(path):
    try:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image from {path}")
            
        image = cv2.resize(image, (105, 105))  # Match original model input size
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=-1)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise
