import yaml
from typing import Any, Dict

def load_yaml(filename: str) -> Dict[str, Any]:
    """
    Load a YAML file and return the contents as a dictionary.

    Args:
        filename (str): The path to the YAML file to be loaded.

    Returns:
        Dict[str, Any]: A dictionary representing the contents
          of the YAML file.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data