import tensorflow as tf


def dataset2array(dataset):
    """Converts a Tensorflow dataset into an array of tensors."""
    
    result = []
    for item in dataset.enumerate():
        result.append(item)
    
    return result


def smoteify(dataset):
    """
    Applies SMOTE to a given dataset.
    Warning: Will read the entire dataset into memory!
    dataset (tf.data.Dataset): The dataset to apply SMOTE to.
    
    Returns:
    tf.data.Dataset: The SMOTEified dataset.
    """
    
    raise Exception("smotify isn't implemented yet")
