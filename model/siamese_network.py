import tensorflow as tf

def get_siamese_model():
    input_shape = (105, 105, 1)
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)

    shared_conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (10,10), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (7,7), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (4,4), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, (4,4), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='sigmoid')
    ])

    encoded_a = shared_conv(input_a)
    encoded_b = shared_conv(input_b)
    L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_a, encoded_b])
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)
    model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=prediction)
    return model