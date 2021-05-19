import io
import json

import tensorflow as tf

from ..io.settings import settings_get
from ..glove.glove import GloVe
from .CategoryCalculator import CategoryCalculator


class TweetsData(tf.data.Dataset):
    """Converts a jsonl tweets file into tensors."""
    
    def _generator():
        settings = settings_get()
        glove = GloVe(settings.data.paths.glove)
        cats = CategoryCalculator()
        reader = io.open(settings.data.paths.input, "r")
        
        for line in reader:
            obj = json.loads(line)
            text = obj["text"].strip()
            
            yield (
                tf.constant(
                    tf.keras.preprocessing.sequence.pad_sequences(
                        glove.embeddings(text),
                        dtype="float32",
                        padding="post",
                        maxlen=settings.data.sequence_length
                    ),
                    dtype="float32"
                ),
                tf.constant(
                    cats.get_category_name(text),
                    dtype="float32"
                )
            )
    
    
    def __new__(this_class):
        """Returns a new tensorflow dataset object."""
        settings = settings_get()
        
        return tf.data.Dataset.from_generator(
            this_class._generator,
            args = ( )
        ).batch(settings.train.batch_size).prefetch(tf.data.AUTOTUNE)
