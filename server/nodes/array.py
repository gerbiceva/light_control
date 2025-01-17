from datatypes import Array, Int, node, Float
import jax.numpy as jnp

@node
def zeros_array(size: Int) -> Array:
    """
    Zeros

    Makes array of zeros of lenght size.

    Parameters
    ----------
    size : Int
        Size of array.

    Returns
    -------
    array : Array
        Array of zeros.
    """
    return jnp.zeros((size))

@node
def ones_array(size: Int) -> Array:
    """
    Ones

    Makes array of ones of lenght size.

    Parameters
    ----------
    size : Int
        Size of array.

    Returns
    -------
    array : Array
        Array of ones.
    """
    return jnp.ones((size))

@node
def ns_array(size: Int, n: Float) -> Array:
    """
    Ns

    Makes array of n of lenght size.

    Parameters
    ----------
    size : Int
        Size of array.

    n : Float
        N in array.

    Returns
    -------
    array : Array
        Array of ns.
    """
    return jnp.ones((size)) * n

