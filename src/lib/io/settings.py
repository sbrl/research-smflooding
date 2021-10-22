import os
import sys
import toml

from loguru import logger
from ..polyfills.dictionary import merge, make_namespace

settings = None


def read_settings_toml(filepath_default: str, filepath_custom: str, overrides):
	"""
	Reads settings from 2 toml files.
	1 file for the default settings, and another for the custom settings which
	override the default settings.
	filepath_default (str): The path the default settings file.
	filepath_custom (str): The path to the custom override settings file.
	"""
	
	settings_default = toml.load(filepath_default)
	settings_custom = toml.load(filepath_custom)
	
	merge(settings_custom, settings_default)
	
	if overrides is not None:
		merge(overrides, settings_default)
	
	settings_default["source"] = toml.dumps(settings_default)
	logger.debug(f"[DEBUG] source:\n" + settings_default["source"])
	return make_namespace(settings_default)


def settings_get():
	"""Returns the settings object. You should call load() first."""
	global settings
	
	return settings


def settings_load(filepath_custom: str, filename_default: str = None, **overrides):
	"""
	Loads the settings from the specified custom config file.
	Said settings are applied over the default config file and saving them to
	be imported later.
	filepath_custom (str): The path to the custom config file to load.
	"""
	global settings
	
	if filename_default == None:
		filename_default = "settings.default.toml"
	
	filepath_default = os.path.join(
		os.path.dirname(os.path.dirname(os.path.dirname(__file__))),    # src/
		filename_default
	)
	
	logger.info(f"[settings_load] filepath_custom={filepath_custom}, filepath_default={filepath_default}")
	
	settings = read_settings_toml(filepath_default, filepath_custom, overrides)
