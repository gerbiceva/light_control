from datatypes import node, Array
import jax.numpy as jnp

@node
def add(a: Array, b: Array) -> Array:
    """
    Add

    Adds two arrays.

    Parameters
    ----------
    a : Array
        The first array.
    b : Array
        The second array.

    Returns
    -------
    result : Array
    """
    return a + b


@node
def subtract(a: Array, b: Array) -> Array:
    """
    Subtract

    Subtracts the second array from the first.

    Parameters
    ----------
    a : Array
        The array to subtract from.
    b : Array
        The array to subtract.

    Returns
    -------
    result : Array
    """
    return a - b


@node
def multiply(a: Array, b: Array) -> Array:
    """
    Multiply

    Multiplies two numbers.

    Parameters
    ----------
    a : Array
        The first array.
    b : Array
        The second array.

    Returns
    -------
    result : Array
    """
    return a * b


@node
def divide(a: Array, b: Array) -> Array:
    """
    Divide
    
    Divides the dividend by the divisor.

    Parameters
    ----------
    a : Array
        The dividend of the division.
    b : Array
        The divisor of the division.

    Returns
    -------
    result : Array
    """
    return a / b

@node
def clamp(value: Array, min_value: Array, max_value: Array) -> Array:
    """
    Clamp

    Clamps a value between a minimum and a maximum.

    Parameters
    ----------
    value : Array
        The value to clamp.
    min_value : Array
        The minimum value.
    max_value : Array
        The maximum value.

    Returns
    -------
    result : Array
    """
    return jnp.maximum(min_value, jnp.minimum(value, max_value))
