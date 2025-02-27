import json
import os

class Config:
   def __init__(self, filename: str = 'config.json') -> None:
       self.filename = filename
       self.config = self._load_config()

   def _load_config(self) -> dict:
       try:
           with open(self.filename, 'r') as raw_config:
               return json.load(raw_config)
       except FileNotFoundError:
           raise FileNotFoundError(f"Configuration file not found: {self.filename}")
       except json.JSONDecodeError as e:
           raise json.JSONDecodeError(f"Invalid JSON in configuration file: {self.filename}.  Details: {e}", e.doc, e.pos)


   def get_baseline_name(self) -> str:
       try:
           return self.config['base_model']
       except KeyError:
           raise KeyError("The 'base_model' key is missing from the configuration file.")

   def get(self, key: str, default=None):
       return self.config.get(key, default)

   def __repr__(self):
        return f"Config(filename='{self.filename}', config={self.config})"

if __name__ == '__main__':
   try:
       config_instance = Config('config.json')
       baseline_name = config_instance.get_baseline_name()
       print(f"Baseline model name: {baseline_name}")

       api_key = config_instance.get("api_key")
       print(f"API Key: {api_key}")

       non_existent_setting = config_instance.get("non_existent", "default_value")
       print(f"Non-existent setting: {non_existent_setting}")


   except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
       print(f"Error loading configuration: {e}")
