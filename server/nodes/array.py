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

@node
def array_slice(arr: Array, start: Int, stop: Int) -> Array:
    """
    Slice

    Takes slice of array. One based indexing.

    Parameters
    ----------
    arr : Array
    start : Int
    stop : Int

    Returns
    -------
    arr : Array
    """
    return arr[start-1:stop]

@node
def reverse_array(arr: Array) -> Array:
    """
    Reverse

    Reverse

    Parameters
    ----------
    arr : Array

    Returns
    -------
    arr : Array
    """
    return arr[::-1]

@node
def set_element(arr: Array, which: Int, what: Float) -> Array:
    """
    Set At

    Set element at which to what. One based indexing.

    Parameters
    ----------
    arr : Array
    which : Int
    what : Float

    Returns
    -------
    arr : Array
    """
    return arr.at[which+1].set(what)

@node
def set_slice(arr: Array, where: Int, what: Array) -> Array:
    """
    Set Slice

    Set element at which to what. One based indexing.

    Parameters
    ----------
    arr : Array
    where : Int
    what : Array

    Returns
    -------
    arr : Array
    """

    # Calculate the bounds for the insertion
    start = max(0, where - len(what) // 2)
    end = start + len(what)

    # Ensure bounds don't exceed main_array's size
    if end > len(arr):
        end = len(arr)
        start = end - len(what)

    # Update the main array with the smaller array values
    return arr.at[start:end].set(what[:end - start])