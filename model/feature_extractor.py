import tensorflow as tf

def get_embedding(image):
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image[0].numpy()  # Just a placeholder - use real feature extraction