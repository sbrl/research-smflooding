import tensorflow as tf


def load_image(filepath, image_size, model_type="resnet"):
    
    image = tf.keras.utils.load_img(
        filepath,
        target_size=(image_size, image_size),
        color_mode="rgb"
    )
    
    
    if image_size < 32 and model_type == "resnet":
        image = image.resize((image_size, image_size))
    
    
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    
    if model_type == "resnet":
        image = tf.keras.applications.resnet50.preprocess_input(
            image,
            data_format="channels_last"
        )
    else:
        image = image.div(255)
    
    return image
