import io
import logging


def summarylogger(model):
    """
    Writes the summary for a model with the default logging context.
    model (tf.keras.Model): The model to generate the summary from.
    """
    
    def handle_line(line: str):
        logging.info(line)
    
    model.summary(print_fn=handle_line)
    

def summarywriter(model, filepath_output):
    """
    Writes the summary for a model to a file in the specified location.
    model (tf.keras.Model): The model to generate the summary from.
    filepath_output (str):  The path to the file to write the summary to.
    """
    handle = io.open(filepath_output, "w")
    
    def handle_line(line: str):
        handle.write(f"{line}\n")
    
    model.summary(print_fn=handle_line)
    
    handle.close()
