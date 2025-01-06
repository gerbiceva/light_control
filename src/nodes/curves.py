from datatypes import Curve, Array, node
import jax
import jax.numpy as jnp
from jax import lax

@node
def sample_curve(curve: Curve, x: float) -> float:
    """
    Sample a Curve at a Specific Point

    Samples a given curve function at a specific x-coordinate.

    Parameters
    ----------
    curve : Curve
        A callable curve function that takes a float as input and returns a float.
    x : Float
        The x-coordinate at which to sample the curve.

    Returns
    -------
    curve : Float
        The value of the curve at the specified x-coordinate.
    """
    return curve(x)

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
    return jax.vmap(curve, in_axes=0, out_axes=0)(jnp.linspace(0, 1, resolution))

@node
def pad_curve(curve: Curve, padding: float) -> Curve:
    """
    Add Padding to a Curve

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
    def new_curve(x: float) -> float:
        return lax.cond(
            x < (1 - padding),
            lambda _: curve(x / (1 - padding)),
            lambda _: 0.0,
            operand=None,
        )

    return new_curve

@node
def move_curve(curve: Curve, move: float) -> Curve:
    """
    Shift a Curve Horizontally

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
    def new_curve(x: float) -> float:
        return curve((x + move) % 1)

    return new_curve

@node
def linear_curve() -> Curve:
    """
    Create a Linear Curve

    Creates a curve that represents a linear function y = x.

    Returns
    -------
    curve : Curve
        A linear curve function.
    """
    def new_curve(x: float) -> float:
        return x

    return new_curve

@node
def sin_curve(peaks: float) -> Curve:
    """
    Create a Sine Wave Curve

    Creates a curve that represents a sine wave with a specified number of peaks.

    Parameters
    ----------
    peaks : Float
        The number of sine wave peaks over the interval [0, 1].

    Returns
    -------
    curve : Curve
        A sine wave curve function.
    """
    def new_curve(x: float) -> float:
        return jnp.sin(x * peaks * jnp.pi)

    return new_curve