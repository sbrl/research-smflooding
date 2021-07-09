import tensorflow as tf
import imblearn


def dataset2array(dataset, flip=True):
    """Converts a Tensorflow dataset into an array of tensors."""
    
    data_in = []
    data_out = []
    
    for item in dataset.enumerate():
        if flip:
            data_in.append(item[1])
            data_out.append(item[0])
        else:
            data_in.append(item[0])
            data_out.append(item[1])
    
    return data_in, data_out


def smoteify(dataset):
    """
    Applies SMOTE to a given dataset.
    Warning: Will read the entire dataset into memory!
    dataset (tf.data.Dataset): The dataset to apply SMOTE to.
    
    Returns
    tf.data.Dataset: The SMOTEified dataset.
    """
    
    raise Exception("Not implemented yet - imblearn doesn't work with our dataset dimensions :-(")
    data_in, data_out = dataset2array(dataset)
    
    smoteified_generator = imblearn.keras.balanced_batch_generator(
        data_in, data_out
    )
    
    first_in, first_out = smoteified_generator()
    
    return tf.data.Dataset.from_generator(smoteified_generator)
