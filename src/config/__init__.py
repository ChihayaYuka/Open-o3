import json

class config:
    def __init__(self, filename:str = 'config.json') -> None:
        with open(filename, 'r') as raw_config:
            self.config = json.loads(raw_config.read())

    def get_baseline_name(self) -> str:
        return self.config['base_model']