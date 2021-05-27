import io


def write_file_sync(filepath, content):
    """
    Writes the specified string to disk.
    filepath (str): The filepath to write the string to.
    content (str):  The content to writ  tot he specified file.
    """
    handle = io.open(filepath, "w")
    handle.write(content)
    handle.close()
