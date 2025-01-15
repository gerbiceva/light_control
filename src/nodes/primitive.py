from typing import Callable
from datatypes import Int, Float, String, node, primitive, Color, Curve

@node
@primitive
def make_Int(a: Int) -> Callable:
    """
    Int

    Makes an Int.

    Parameters
    ----------
    number : Int

    Returns
    -------
    Callable
    """
    def integer() -> Int:
        """
        Int

        Makes an Int.

        Parameters
        ----------
        None

        Returns
        -------
        Int : Int
        """
        return a

    return integer

@node
@primitive
def make_Float(a: Float) -> Callable:
    """
    Float

    Makes an Float.

    Parameters
    ----------
    number : Float

    Returns
    -------
    Callable
    """
    def floating_point() -> Float:
        """
        Float

        Makes an Float.

        Parameters
        ----------
        None

        Returns
        -------
        Float : Float
        """
        return a

    return floating_point

@node
@primitive
def make_String(a: String) -> Callable:
    """
    String

    Makes an String.

    Parameters
    ----------
    String : String

    Returns
    -------
    Callable
    """
    def string() -> String:
        """
        String

        Makes an String.

        Parameters
        ----------
        None

        Returns
        -------
        String : String
        """
        
        return a
    return string

@node
@primitive
def make_Color(a: Color) -> Callable:
    """
    Color

    Makes an Color.

    Parameters
    ----------
    color : Color

    Returns
    -------
    Callable
    """
    def color() -> Color:
        """
        Color

        Makes an Color.

        Parameters
        ----------
        None

        Returns
        -------
        Color : Color
        """
        return a

    return color

@node
@primitive
def make_Curve(a: Curve) -> Callable:
    """
    Curve

    Makes a Curve.

    Parameters
    ----------
    curve : Curve

    Returns
    -------
    Callable
    """
    def curve() -> Color:
        """
        Curve

        Makes an Curve.

        Parameters
        ----------
        None

        Returns
        -------
        Curve : Curve
        """
        return a

    return curve