from datatypes import node, Array, Float, Int
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

@node
def add_scalar(a: Array, b: Float) -> Array:
    """
    Add Scalar

    Adds an array and scalar.

    Parameters
    ----------
    a : Array
        The array.
    b : Float
        The scalar.

    Returns
    -------
    result : Array
    """
    return a + b


@node
def subtract_scalar(a: Array, b: Float) -> Array:
    """
    Subtract Scalar

    Subtracts the scalar from the array.

    Parameters
    ----------
    a : Array
        The array to subtract from.
    b : Float
        The scalar to subtract.

    Returns
    -------
    result : Array
    """
    return a - b


@node
def multiply_scalar(a: Array, b: Float) -> Array:
    """
    Multiply Scalar

    Multiplies array with scalar.

    Parameters
    ----------
    a : Array
        The array.
    b : Float
        The scalar.

    Returns
    -------
    result : Array
    """
    return a * b


@node
def divide_scalar(a: Array, b: Array) -> Array:
    """
    Divide Scalar
    
    Divides the array with the scalar.

    Parameters
    ----------
    a : Array
        The dividend array.
    b : Float
        The divisor of the array.

    Returns
    -------
    result : Array
    """
    return a / b

@node
def clamp_scalar(value: Array, min_value: Float, max_value: Float) -> Array:
    """
    Clamp with scalars

    Clamps a value between a minimum and a maximum.

    Parameters
    ----------
    value : Array
        The value to clamp.
    min_value : Float
        The minimum value.
    max_value : Float
        The maximum value.

    Returns
    -------
    result : Array
    """
    return jnp.maximum(min_value, jnp.minimum(value, max_value))

@node
def interpolate(array: Array, size: Int):
    """
    Interpolate
    
    Interpolates array from one size to another.

    Parameters
    ----------
    value : Array
        The value to clamp.
    size : Int

    Returns
    -------
    result : Array
    """
    return jnp.interp(jnp.linspace(0, array.size-1, size), jnp.arange(array.size), array)

@node
def average(array: Array):
    """
    Average
    
    Interpolates array from one size to another.

    Parameters
    ----------
    value : Array
        The value to clamp.

    Returns
    -------
    result : Float
    """
    return jnp.average(array)