import tensorflow as tf
import imblearn


def dataset2array(dataset):
    """Converts a Tensorflow dataset into an array of tensors."""
    
    data_in = []
    data_out = []
    
    for item in dataset.enumerate():
        data_in.append(item[1])
        data_out.append(item[0])
    
    return data_in, data_out


def smoteify(dataset):
    """
    Applies SMOTE to a given dataset.
    Warning: Will read the entire dataset into memory!
    dataset (tf.data.Dataset): The dataset to apply SMOTE to.
    
    Returns:
    tf.data.Dataset: The SMOTEified dataset.
    """
    
    data_in, data_out = dataset2array(dataset)
    
    print("FIRST IN")
    tf.print(data_in[0])
    print("FIRST OUT")
    tf.print(data_out[0])
    
    smoteified_generator = imblearn.keras.balanced_batch_generator(
        data_in, data_out
    )
    
    
    return tf.data.Dataset.from_generator(smoteified_generator)
