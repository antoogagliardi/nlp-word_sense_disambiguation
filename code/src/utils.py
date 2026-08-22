import ast
import yaml


# Read configuration file
def read_config_file(config_path:str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg



def load_json_data(data_path:str) -> list:
    """
    Opens the .json file containing the data and extracts its contents.

    Args:
        data_path (str): Path to the input file.
    Returns:
        list: Content of the input file encapsulated in a list.
    """

    # Read the data from the JSON file
    with open(data_path) as f:
        data = f.read()
    # Reconstruct the data as a dictionary
    result = ast.literal_eval(data)

    return result