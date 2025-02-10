import inspect
import importlib
import os
from numpydoc.docscrape import FunctionDoc

def load_nodes(directory: str) -> (list, list, list):
    # List to store the imported modules
        modules = []
        namespaces = []

        # Iterate over all .py files in the directory
        for filename in os.listdir(directory):
            if (
                filename.endswith(".py")
                and filename != "__init__.py"
                and not filename.startswith("_")
            ):
                module_name = filename[:-3]  # Remove the .py extension
                module = importlib.import_module(f"{directory}.{module_name}")
                modules.append(module)
                namespaces.append(module_name)

        nodes = []
        threads = []
        each_tick = []
        for namespace, module in zip(namespaces, modules):
            for member in inspect.getmembers(module):
                if (
                    member[0][0] != "_"
                    # and inspect.isfunction(member[1])
                    and hasattr(member[1], "__is_node__")
                ):
                    nodes.append((member[1], FunctionDoc(member[1]), namespace))
                if hasattr(member[1], "__each_tick__"):
                    each_tick.append(member[1])
                if hasattr(member[1], "__thread__"):
                    threads.append(member[1])
                            
        return {f"{node[2]}/{node[1]['Summary'][0]}": node[0] for node in nodes}, threads, each_tick