import inspect
from numpydoc.docscrape import FunctionDoc
import importlib
import os



# def aaa():
#     """
#     Create a ColorArray with Zero Values

#     Creates a color array of size `n` with all values initialized to zero.

#     Parameters
#     ----------
#     n : Int
#         The number of colors (columns) in the color array.

#     Returns
#     -------
#     out : ColorArray
#         A color array with three rows (for hue, saturation, and brightness) 
#         and `n` columns, all initialized to zero.
#     outt : ColorAdsdss
#         Asadsadsa array with three rows (for hue, saturation, and brightness) 
#         and `n` columns, all initialized to zero.
#     """
#     pass

# neki = FunctionDoc(aaa)

# print(neki["Extended Summary"])

# Path to the directory containing the files
# directory = "nodes"

# # List to store the imported modules
# modules = []

# # Iterate over all .py files in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".py") and filename != "__init__.py" and not filename.startswith("_"):
#         module_name = filename[:-3]  # Remove the .py extension
#         module = importlib.import_module(f"{directory}.{module_name}")
#         modules.append(module)

# nodes = []
# for module in modules:
#     for member in inspect.getmembers(module):
#         if member[1].__doc__ and member[0][0] != '_' and inspect.isfunction(member[1]):
#             nodes.append((member[0], parse(member[1].__doc__)))
#             print(nodes[-1][1].params[0].arg_name)

# print(nodes)