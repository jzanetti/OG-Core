
import numpy as np
from json import load as json_load

def get_default_params(file_path = "ogcore/default_parameters.json"):

    with open(file_path, "r") as f:
        default_p = json_load(f)
    
    return default_p


