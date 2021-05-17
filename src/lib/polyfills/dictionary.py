

def merge(source, target):
    """
    Merges 2 dictionaries.
    CAUTION: The target will be mutated!
    source (dict): The source dictionary to read from.
    target (dict): The target to apply the source to.
    """
    
    for key in source:
        if isinstance(source[key], dict) and isinstance(target[key], dict):
            merge(source[key], target[key])
        else:
            target[key] = source[key]
