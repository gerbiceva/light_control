from datatypes import ColorArray, node, Int
import jax.numpy as jnp

@node
def color_array(n: Int) -> ColorArray:
    """
    Create a ColorArray with Zero Values

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