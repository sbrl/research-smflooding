import os

# Ref https://stackoverflow.com/a/61733714/1460422

###
## Environment parsing and validation helpers
## 		@sbrl, Licence: AGPLv3
###

## Changelog:
# 2024-11-14: Fix crash on line #107 unterminated string literal
# 2024-09-29: Create this changelog, prepare for reuse

##############################################################################

# Simple polyfill for Symbol from JS: https://devdocs.io/javascript/global_objects/symbol

class Symbol:
	def __init__(self, name=''):
		self.name = f"Symbol({name})"

	def __repr__(self):
		return self.name

##############################################################################


SYM_RAISE_EXCEPTION = Symbol("__env_read_raise_exception")

envs_read = []


def read(name, type_class, default=SYM_RAISE_EXCEPTION):
	"""
	Reads, parses, and returns an environment variable with the specified name and type, with an optional default value.

	If the environment variable does not exist and no default value is provided, an `Exception` is raised. Otherwise, the environment variable value is converted to the specified type and returned.
	
	If `type_class == bool` and no `default` value is provided, then the default value is set to `False` and an `Exception` is **not** raised.
	
	Args:
	name (str): The name of the environment variable to read.
	type_class (type): The type to convert the environment variable value to.
	default (Any, optional): The default value to use if the environment variable does not exist. Defaults to `SYM_RAISE_EXCEPTION`, which will raise an exception if the variable does not exist.

	Returns:
			Any: The environment variable value converted to the specified type.

	Raises:
			Exception: If the environment variable does not exist and no default value is provided.
	"""

	if name not in os.environ:
		if type_class == bool and default == SYM_RAISE_EXCEPTION:
			default = False
		if default == SYM_RAISE_EXCEPTION:
			raise Exception(f"Error: Environment variable {name} does not exist")
		envs_read.append([name, default, True])
		return default
	
	result = os.environ[name]
	if type_class == bool:
		result = False if default == True else True
	else:
		result = type_class(result)
	
	envs_read.append([name, result, False])
	return result


def print_all(table=True):
	"""
	Prints a formatted table of all environment variables that have been read so far.

	The table includes the name, type, value, and flags of each environment variable. The column widths are automatically adjusted to fit the longest values.

	If no environment variables have been read yet, a message is printed indicating that.
	
	Args:
	table (bool): Whether to print in a table or not. Defaults to True. If False, then values are directly printed in a list instead of in a pretty table.
	"""

	if not envs_read:
		print("No environment variables have been read yet.")
		return
	
	
	# Calculate column widths
	width_name = max(len("Name"), max(len(env[0]) for env in envs_read))
	width_type = max(len("Type"), max(
		len(type(env[1]).__name__) for env in envs_read))
	width_value = max(len("Value"), max(len(str(env[1])) for env in envs_read))
	width_flags = max(len("Flags"), len("default"))
	
	
	
	if not table:
		print("===================================")
		for env in envs_read:
			key, value, is_default = env
			prefix = "* " if is_default else ""
			print(f"> {key.ljust(width_name)} {value}")
		print(f"Total {len(envs_read)} values")
		print("===================================")
	
	
	# Create the table format string
	format_string = f"| {{:<{width_name}}} | {{:<{width_type}}} | {{:<{width_value}}} | {{:<{width_flags}}} |"

	# Calculate total width
	total_width = width_name + width_type + width_value + \
		width_flags + 13  # 13 accounts for separators and spaces

	# Print the header
	print("+" + "-" * (total_width - 2) + "+")
	print(format_string.format("Name", "Type", "Value", "Flags"))
	print("+" + "=" * (width_name + 2) + "+" + "=" * (width_type + 2) +
		  "+" + "=" * (width_value + 2) + "+" + "=" * (width_flags + 2) + "+")

	# Print each environment variable
	for name, value, is_default in envs_read:
		flags = "default" if is_default else ""
		print(format_string.format(name, type(value).__name__, str(value), flags))
		print("+" + "-" * (width_name + 2) + "+" + "-" * (width_type + 2) +
			  "+" + "-" * (width_value + 2) + "+" + "-" * (width_flags + 2) + "+")


def val_exists(value, msg_error="The file or directory () does not exist, or I don't have permission to read it"):
	if not os.path.exists(value):
		raise Exception(msg_error.replace("()", f"'{value}'"))


def val_file_exists(value, msg_error="The file () does not exist, or I don't have permission to read it"):
	if not os.path.isfile(value):
		raise Exception(msg_error.replace("()", f"'{value}'"))


def val_dir_exists(value, msg_error="The directory () does not exist, or I don't have permission to read it", create=False):
	if not os.path.isdir(value):
		if create:
			if os.path.exists(value):
				raise Exception(f"Attempted to create directory '{value}', but it already exists and is not a directory")
			os.makedirs(value)
			return
		raise Exception(msg_error.replace("()", f"'{value}'"))
