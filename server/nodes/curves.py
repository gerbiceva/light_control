from datatypes import Curve, Array, node, String
import jax.numpy as jnp
from sympy import symbols, lambdify, sympify, Piecewise, sin

x = symbols("x")

@node
def sample_curve(curve: Curve, number: float) -> float:
    """
    Sample a Curve at a Specific Point

    Samples a given curve function at a specific x-coordinate.

    Parameters
    ----------
    curve : Curve
        A callable curve function that takes a float as input and returns a float.
    number : Float
        The x-coordinate at which to sample the curve.

    Returns
    -------
    curve : Float
        The value of the curve at the specified x-coordinate.
    """
    return lambdify(x, curve, 'jax')(number)

@node
def sample_array(curve: Curve, resolution: int) -> Array:
    """
    Sample a Curve over an Array

    Samples a given curve function over a specified resolution, producing an array of values.

    Parameters
    ----------
    curve : Curve
        A callable curve function that takes a float as input and returns a float.
    resolution : Int
        The number of equally spaced samples to take along the curve.

    Returns
    -------
    array : Array
        An array of sampled values from the curve.
    """
    return lambdify(x, curve, "jax")(jnp.linspace(0, 1, resolution))

@node
def pad_curve(curve: Curve, padding: float) -> Curve:
    """
    Pad Curve

    Modifies a curve by adding a padding region at the end where the curve evaluates to zero.

    Parameters
    ----------
    curve : Curve
        A callable curve function that takes a float as input and returns a float.
    padding : Float
        The proportion of the curve to pad with zeros at the end.

    Returns
    -------
    curve : Curve
        A new curve with the specified padding.
    """
    return Piecewise((curve, x < padding), (0, x>=padding))

@node
def shift_curve(curve: Curve, move: float) -> Curve:
    """
    Shift Curve

    Modifies a curve by shifting it horizontally by a specified amount, wrapping around at the boundaries.

    Parameters
    ----------
    curve : Curve
        A callable curve function that takes a float as input and returns a float.
    move : Float
        The amount to shift the curve, in the range [0, 1].

    Returns
    -------
    curve : Curve
        A new curve shifted by the specified amount.
    """

    return curve.subs(x, x + move)

@node
def linear_curve() -> Curve:
    """
    Linear Curve

    Creates a curve that represents a linear function y = x.

    Returns
    -------
    curve : Curve
        A linear curve function.
    """
    return x

@node
def curve(string: String):
    """
    String 2 Curve

    Creates a curve from an arbitrary string.

    Parameters
    ----------
    string : String
        Converts a string to a curve.

    Returns
    -------
    curve : Curve
        A string representing a curve.
    """
    return sympify(string)

@node
def sin_curve() -> Curve:
    """
    Sine Wave

    Creates a curve that represents a sine wave with a specified number of peaks.

    Returns
    -------
    curve : Curve
        A sine wave curve function.
    """
    return sin(x)

@node
def sub_var(curve, name, value):
    """
    Sub Value

    Substitute value in curve.

    Parameters
    ----------
    curve : Curve
    name : String
        Converts a string to a curve.
    value : Float

    Returns
    -------
    curve : Curve
        A string representing a curve.
    """
    return curve.subs(name, value)