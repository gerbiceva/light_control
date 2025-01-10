import inspect
import importlib
import os
from numpydoc.docscrape import FunctionDoc

def load_nodes():
    # Path to the directory containing the files
    directory = "nodes"

    # List to store the imported modules
    modules = []

    # Iterate over all .py files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py" and not filename.startswith("_"):
            module_name = filename[:-3]  # Remove the .py extension
            module = importlib.import_module(f"{directory}.{module_name}")
            modules.append(module)

    nodes = []
    for module in modules:
        for member in inspect.getmembers(module):
            if member[0][0] != '_' and inspect.isfunction(member[1]) and hasattr(member[1], "__is_node__"):
                nodes.append((member[1], FunctionDoc(member[1])))

    return nodes