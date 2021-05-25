# / removed from being the last param, ref https://stackoverflow.com/a/56514307
def removeprefix(self: str, prefix: str, /) -> str:
    """
    Removes a prefix from a string.
    Polyfills string.removeprefix(), which is introduced in Python 3.9+.
    Ref https://www.python.org/dev/peps/pep-0616/#specification
    """
    if self.startswith(prefix):
        return self[len(prefix):]
    else:
        return self[:]


# / removed from being the last param, ref https://stackoverflow.com/a/56514307
def removesuffix(self: str, suffix: str) -> str:
    """
    Removes a suffix from a string.
    Polyfills string.removesuffix(), which is introduced in Python 3.9+.
    Ref https://www.python.org/dev/peps/pep-0616/#specification
    """
    # suffix='' should not call self[:-0].
    if suffix and self.endswith(suffix):
        return self[:-len(suffix)]
    else:
        return self[:]
