from datatypes import ColorArray, node, Int
import jax.numpy as jnp

@node
def zero_color_array(n: Int) -> ColorArray:
    """
    Zero Color Array

    Creates a color array of size `n` with all values initialized to zero.

    Parameters
    ----------
    n : Int
        The number of colors (columns) in the color array.

    Returns
    -------
    hsv : ColorArray
        A color array with three rows (for hue, saturation, and brightness) 
        and `n` columns, all initialized to zero.
    """
    return jnp.zeros((3, n))

@node
def color_array(n: Int) -> ColorArray:
    """
    Color Array

    Creates a color array of size `n` with brightness and saturation to one.

    Parameters
    ----------
    n : Int
        The number of colors (columns) in the color array.

    Returns
    -------
    hsv : ColorArray
        A color array with three rows (for hue, saturation, and brightness) 
        and `n` columns, all initialized to zero.
    """
    return jnp.ones((3, n))