import numpy as np
import pandas as pd
import os

def get_shape(obj):
    """
    Recursively infer shape of nested structure containing lists, dicts, and numpy scalars.
    """
    # If dict, assume data is stored under "value"
    if isinstance(obj, dict):
        if "value" in obj:
            return get_shape(obj["value"])
        else:
            return ()  # no useful data to inspect

    # If numpy array: return its native shape
    if isinstance(obj, np.ndarray):
        return obj.shape

    # If list or tuple: infer shape from first element
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return (0,)
        return (len(obj),) + get_shape(obj[0])

    # Base case: scalar → no more dimensions
    return ()

def get_param(p, save_dir):
    rows = []
    for name, entry in p._data.items():
        if name == "e":
            x = 3
        try:
            proc_shape = getattr(p, name).shape
            proc_value = getattr(p, name).mean()
        except:
            proc_shape = None
            proc_value = getattr(p, name)
        rows.append({
            "name": name,
            "description": entry.get("description"),
            "shape": proc_shape,
            "value": proc_value
            #"label": entry.get("label"),
            #"type": entry.get("ptype")
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "OG-Core_nz_parameters.csv"), index=False)


