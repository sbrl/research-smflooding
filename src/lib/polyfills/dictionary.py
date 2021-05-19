from types import SimpleNamespace


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


def make_namespace(dictionary):
	"""
	Converts a dictionary to a namespace to allow for dotted.syntax.
	** = spread operator, I think
	Ref https://stackoverflow.com/a/36908/1460422
	"""
	
	result = SimpleNamespace(**dictionary)
	
	for key in dictionary:
		if dictionary[key] is dict:
			result[key] = make_namespace(**dictionary[key])
	
	return result
