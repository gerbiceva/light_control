from datatypes import Curve, Array, node, String, Curve2D, Float, Int, Array2D
import jax.numpy as jnp
from sympy import lambdify, sympify, Piecewise, sin, MatrixSymbol


@node
def sample_curve_2D(curve: Curve2D, x: Float, y: Float) -> Float:
    """
    Sample a Curve at a Specific Point

    Samples a given curve function at a specific x,y-coordinate.

    Parameters
    ----------
    curve : Curve2D
        A callable curve function that takes a float as input and returns a float.
    x : Float
    y : Float
        
    Returns
    -------
    t : Float
        The value of the curve at the specified x-coordinate.
    """
    return lambdify(x, curve, 'jax')(x, y)

@node
def sample_array_2D(curve: Curve2D, h: Int, w: Int) -> Array2D:
    """
    Sample Curve 2D

    Samples a given curve function over a specified resolution, producing an array of values.

    Parameters
    ----------
    curve : Curve2D
        A callable curve function that takes a float as input and returns a float.
    h : Int
        The number of equally spaced samples to take along the curve.
    w : Int

    Returns
    -------
    array : Array2D
        An array of sampled values from the curve.
    """
    # Convert the expression to a JAX-compatible function
    jax_func = lambdify((sympify("x"), sympify("y")), curve, 'jax')

    # Create a 2D grid of values
    x_vals = jnp.linspace(-1, 1, h)  # 100 points between 0 and π
    y_vals = jnp.linspace(-1, 1, w)
    X, Y = jnp.meshgrid(x_vals, y_vals)

    # Evaluate the function on the entire grid
    return jax_func(X, Y)

# @node
# def pad_curve(curve: Curve, padding: float) -> Curve:
#     """
#     Pad Curve

#     Modifies a curve by adding a padding region at the end where the curve evaluates to zero.

#     Parameters
#     ----------
#     curve : Curve
#         A callable curve function that takes a float as input and returns a float.
#     padding : Float
#         The proportion of the curve to pad with zeros at the end.

#     Returns
#     -------
#     curve : Curve
#         A new curve with the specified padding.
#     """
#     return Piecewise((curve, x < padding), (0, x>=padding))

# @node
# def shift_curve(curve: Curve, move: float) -> Curve:
#     """
#     Shift Curve

#     Modifies a curve by shifting it horizontally by a specified amount, wrapping around at the boundaries.

#     Parameters
#     ----------
#     curve : Curve
#         A callable curve function that takes a float as input and returns a float.
#     move : Float
#         The amount to shift the curve, in the range [0, 1].

#     Returns
#     -------
#     curve : Curve
#         A new curve shifted by the specified amount.
#     """

#     return curve.subs(x, x + move)

# @node
# def linear_curve() -> Curve:
#     """
#     Linear Curve

#     Creates a curve that represents a linear function y = x.

#     Returns
#     -------
#     curve : Curve
#         A linear curve function.
#     """
#     return x

@node
def curve2D(string: String):
    """
    String 2 Curve 2D

    Creates a curve from an arbitrary string.

    Parameters
    ----------
    string : String
        Converts a string to a curve.

    Returns
    -------
    curve : Curve2D
        A string representing a curve.
    """
    return sympify(string)

# @node
# def sin_curve() -> Curve:
#     """
#     Sine Wave

#     Creates a curve that represents a sine wave with a specified number of peaks.

#     Returns
#     -------
#     curve : Curve
#         A sine wave curve function.
#     """
#     return sin(x)