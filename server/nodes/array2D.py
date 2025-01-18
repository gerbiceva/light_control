from datatypes import Array2D, Int, node, Float, Array
import jax.numpy as jnp
import jax
import time

@node
def zeros2D(h: Int, w: Int) -> Array2D:
    """
    Zeros 2D

    Makes array of zeros of size h,w.

    Parameters
    ----------
    h : Int
    w : Int

    Returns
    -------
    array : Array2D
        Array of zeros.
    """
    return jnp.zeros((h, w))

@node
def ones2D(h: Int, w: Int) -> Array2D:
    """
    Ones 2D

    Makes an array of ones of size h,w.

    Parameters
    ----------
    h : Int
    w : Int

    Returns
    -------
    array : Array2D
        Array of ones.
    """
    return jnp.ones((h, w))

@node
def ns2D(h: Int, w: Int, n: Float) -> Array2D:
    """
    Ns 2D

    Makes array of n of size h, w.

    Parameters
    ----------
    h : Int

    w : Int

    n : Float

    Returns
    -------
    array : Array2D
        Array of ns.
    """
    return jnp.ones((h, w)) * n

@node
def flatten(arr: Array2D) -> Array:
    """
    Array2D 2 Array

    Ravels an array.

    Parameters
    ----------
    array2D : Array2D

    Returns
    -------
    array : Array
    """
    return arr.ravel()

@node
def random_noise(h, w) -> Array2D:
    """
    Random noise 2D

    Creates 2D array with random values from 0-1.

    Parameters
    ----------
    h : Int
    w : Int

    Returns
    -------
    array : Array2D
    """
    return jax.random.uniform(jax.random.PRNGKey(int(time.time()*1000)), shape=(h, w), minval=0, maxval=1)


# @node
# def array_slice(arr: Array, start: Int, stop: Int) -> Array:
#     """
#     Slice 2D

#     Takes slice of array. One based indexing.

#     Parameters
#     ----------
#     arr : Array
#     start : Int
#     stop : Int

#     Returns
#     -------
#     arr : Array
#     """
#     return arr[start-1:stop]

# @node
# def reverse_array(arr: Array) -> Array:
#     """
#     Reverse

#     Reverse

#     Parameters
#     ----------
#     arr : Array

#     Returns
#     -------
#     arr : Array
#     """
#     return arr[::-1]

@node
def set_element(arr: Array2D, x: Int, y: Int, what: Float) -> Array2D:
    """
    Set At 2D

    Set element at x,y to what. One based indexing.

    Parameters
    ----------
    arr : Array2D
    x : Int
    y : Int
    what : Float

    Returns
    -------
    arr : Array2D
    """
    return arr.at[x+1, y+1].set(what)

# @node
# def set_slice(arr: Array, where: Int, what: Array) -> Array:
#     """
#     Set Slice

#     Set element at which to what. One based indexing.

#     Parameters
#     ----------
#     arr : Array
#     where : Int
#     what : Array

#     Returns
#     -------
#     arr : Array
#     """

#     # Calculate the bounds for the insertion
#     start = max(0, where - len(what) // 2)
#     end = start + len(what)

#     # Ensure bounds don't exceed main_array's size
#     if end > len(arr):
#         end = len(arr)
#         start = end - len(what)

#     # Update the main array with the smaller array values
#     return arr.at[start:end].set(what[:end - start])