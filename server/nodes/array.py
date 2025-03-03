from datatypes import Array, Int, node, Float, ColorArray
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
def reverse_array_color(arr: ColorArray) -> ColorArray:
    """
    Reverse Colors

    Reverse

    Parameters
    ----------
    arr : ColorArray

    Returns
    -------
    arr : ColorArray
    """
    return arr[:, ::-1]

@node
def set_element(arr: Array, index: Int, value: Float) -> Array:
    """
    Set At

    Set element at which to what. One based indexing.

    Parameters
    ----------
    arr : Array
    index : Int
    value : Float

    Returns
    -------
    arr : Array
    """
    return arr.at[index+1].set(value)

@node
def get_element(arr: Array, index: Int) -> Array:
    """
    Get At

    Get element at which to what. One based indexing.

    Parameters
    ----------
    arr : Array
    index : Int

    Returns
    -------
    arr : Array
    """
    return arr[index+1]

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

@node
def get_slice(arr: Array, start: Int, stop: Int) -> Array:
    """
    Get Slice

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
def split_2(arr: Array) -> Array:
    """
    Split 2

    Split an array in two.

    Parameters
    ----------
    array : Array

    Returns
    -------
    array1 : Array
    array2 : Array
    """
    return tuple(jnp.array_split(arr, 2))

@node
def split_3(arr: Array) -> Array:
    """
    Split 3

    Split an array in two.

    Parameters
    ----------
    array : Array

    Returns
    -------
    array1 : Array
    array2 : Array
    array3 : Array
    """
    return tuple(jnp.array_split(arr, 3))

@node
def concat(arr1: Array, arr2: Array) -> Array:
    """
    Concat

    Concat two arrays.

    Parameters
    ----------
    array1 : Array
    array2 : Array

    Returns
    -------
    array : Array
    """
    return jnp.concatenate([arr1, arr2])

@node
def set_slice_color(arr: ColorArray, where: Int, what: ColorArray) -> ColorArray:
    """
    Set Color Slice

    Set element at which to what. One based indexing.

    Parameters
    ----------
    arr : ColorArray
    where : Int
    what : ColorArray

    Returns
    -------
    arr : ColorArray
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

@node
def get_slice_color(arr: ColorArray, start: Int, stop: Int) -> ColorArray:
    """
    Get Color Slice

    Takes slice of array. One based indexing.

    Parameters
    ----------
    arr : ColorArray
    start : Int
    stop : Int

    Returns
    -------
    arr : ColorArray
    """
    return arr[start-1:stop]

@node
def split_2_color(arr: ColorArray) -> ColorArray:
    """
    Split 2

    Split an array in two.

    Parameters
    ----------
    array : ColorArray

    Returns
    -------
    array1 : ColorArray
    array2 : ColorArray
    """
    return tuple(jnp.array_split(arr, 2))

@node
def split_3_color(arr: ColorArray) -> ColorArray:
    """
    Split 3

    Split an array in two.

    Parameters
    ----------
    array : ColorArray

    Returns
    -------
    array1 : ColorArray
    array2 : ColorArray
    array3 : ColorArray
    """
    return tuple(jnp.array_split(arr, 3))
    
@node
def concat_color(arr1: ColorArray, arr2: ColorArray) -> ColorArray:
    """
    Concat colors

    Concat two colors.

    Parameters
    ----------
    array1 : ColorArray
    array2 : ColorArray

    Returns
    -------
    array : ColorArray
    """
    return jnp.hstack([arr1, arr2])