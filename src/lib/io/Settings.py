import os.path
import toml
import ..polyfills.dictionary


class Settings:
    """Manges settings from a default and a custom file."""
    
    def __init__(self, filepath_default: str, filepath_custom: str):
        """Initialises a new Settings class instance."""
        settings_default = toml.load(filepath_default)
        settings_custom = toml.load(filepath_custom) if os.path.exists(filepath_custom) else {}
        
        self.settings = dictionary.merge(settings_custom, settings_default)
    
    def get(self, path: str):
        parts = path.split(".")
        
        target = self.settings
        
        for subkey in parts:
            if target[subkey] is None:
                return None
            else:
                target = target[subkey]
        
        return target
