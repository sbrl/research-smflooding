import os
import toml
from types import SimpleNamespace
from ..polyfills.dictionary import merge, make_namespace

settings = None


def read_settings_toml(filepath_default: str, filepath_custom: str):
	"""
	Reads settings from 2 toml files - 1 for the default settings, and another
	for the custom settings which override the default settings.
	filepath_default (str): The path the default settings file.
	filepath_custom (str): The path to the custom override settings file.
	"""
	
	settings_default = toml.load(filepath_default)
	settings_custom = toml.load(filepath_custom)
	
	merge(settings_custom, settings_default)
	
	return make_namespace(settings_default)


def settings_get():
	"""Returns the settings object. You should call load() first."""
	global settings
	
	return settings


def settings_load(filepath_custom: str):
	"""
	Loads the settings from the specified custom config file.
	Said settings are applied over the default config file and saving them to
	be imported later.
	filepath_custom (str): The path to the custom config file to load.
	"""
	global settings
	
	filepath_default = os.path.join(
		os.path.dirname(os.path.dirname(os.path.dirname(__file__))),    # src/
		"settings.default.toml"
	)
	
	settings = read_settings_toml(filepath_default, filepath_custom)
