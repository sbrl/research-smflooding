categories = [
    { "i": 0, "name": "positive" },
    { "i": 1, "name": "negative" }
]

def index2name(i: int):
    """
    Returns the name for a given category index.
    For example, given the integer 0, this function will return the string
    "positive".
    i (int): The category index to convert.
    returns: The name of the category associated with the given index.
    """
    for cat in categories:
        if i == cat["i"]:
            return cat["name"]

def name2index(name: str):
    """
    Returns the index for given category name.
    For example, given the string "positive", this function returns the
    integer 0.
    name (string): The cateegory name to convert.
    returns: The category index associated with the given category name.
    """
    for cat in categories:
        if name == cat["name"]:
            return cat["i"]
