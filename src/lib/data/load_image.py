import tensorflow as tf

from loguru import logger

# Ref https://stackoverflow.com/a/23575424/1460422
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(filepath, image_size, model_type="resnet"):
    image = None
    try:
        image = tf.keras.utils.load_img(
        filepath,
        target_size=(image_size, image_size),
        color_mode="rgb"
        )
    except Exception as e:
        logger.warn(f"Ignoring error: {e}")
        return None
    
    
    if image_size < 32 and model_type == "resnet":
        image = image.resize((image_size, image_size))
    
    
    # image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.keras.preprocessing.image.img_to_array(
        image # Apparently the dtype defaults to float32
    )
    
    
    if model_type == "resnet":
        image = tf.keras.applications.resnet50.preprocess_input(
            image,
            data_format="channels_last"
        )
    else:
        image = image.div(255)
    
    return image
