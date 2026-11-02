"""MobileNetV2 with preprocessing embedded in the serialized model."""


def build_model(num_classes: int, image_size: int = 224, weights="imagenet"):
    import tensorflow as tf

    inputs = tf.keras.Input((image_size, image_size, 3), name="rgb_pixels_0_255")
    x = tf.keras.layers.RandomFlip("horizontal", name="augmentation")(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocessing")(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights=weights, pooling="avg"
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="breed")(x)
    return tf.keras.Model(inputs, outputs, name="cat_vision")
