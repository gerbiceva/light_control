from datatypes import String, Int, Float, Array, node, ColorArray
from utils import print_strip

@node
def print_string(string: String):
    """
    Print String

    Prints a string.

    Parameters
    ----------
    string : String

    Returns
    -------
    None
    """
    print(string)

@node
def print_int(integer: Int):
    """
    Print Integer

    Prints an integer.

    Parameters
    ----------
    integer : Int

    Returns
    -------
    None
    """
    print(integer)

@node
def print_float(flt: Float):
    """
    Print Float

    Prints a floating-point number.

    Parameters
    ----------
    flt : Float

    Returns
    -------
    None
    """
    print(flt)

@node
def print_array(arr: Array):
    """
    Print Array

    Prints an array.

    Parameters
    ----------
    arr : Array

    Returns
    -------
    None
    """
    print(arr)

@node
def print_color_array(color_arr: ColorArray):
    """
    Print Color Array

    Prints a color array.

    Parameters
    ----------
    color_arr : ColorArray

    Returns
    -------
    None
    """
    print_strip(color_arr)

@node
def print_array_shape(arr: Array):
    """
    Print Array Shape

    Prints an array shape.

    Parameters
    ----------
    arr : Array

    Returns
    -------
    None
    """
    print(arr.shape)
