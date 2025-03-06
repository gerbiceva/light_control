from datatypes import Float, Array, node, Int, required
import jax.numpy as jnp

@node
def array_if(cond: Array, a: Array, b: Array) -> Array:
    """
    Array If

    Where test is positive this, where it's negative that.

    Parameters
    ----------
    conditions : Array
        Test array.

    a : Array
        This array.

    b : Array
        That array.

    Returns
    -------
    array : Array
        Output array.
    """
    return jnp.where(cond >= 0, a, b)

@node
def float_pass_if(cond: Int, a: Float, b: Float):
    """
    Passthrough If

    If condition is not 0 passes a, else passes b.

    Parameters
    ----------
    condition : Int
    a : Float
    b : Float

    Returns
    -------
    a : Float
    b : Float
    """
    return (a, None) if cond else (None, b)

@node
def pass_1(input: Int):
    """
    Pass 1

    If input is 1 passes it else nothing.

    Parameters
    ----------
    input : Int

    Returns
    -------
    output : Int
    """
    return 1 if input == 1 else None

@node
@required(1)
def first_float(a,b):
    """
    First float

    Passess forward the first value it gets.

    Parameters
    ----------
    a : Float
    b : Float

    Returns
    -------
    out : Float
    """
    return a if b is None else b

@node
@required(1)
def first_int(a,b):
    """
    First int

    Passess forward the first value it gets.

    Parameters
    ----------
    a : Float
    b : Float

    Returns
    -------
    out : Float
    """
    return a if b is None else b

@node
@required(1)
def first_array(a,b):
    """
    First Array

    Passess forward the first value it gets.

    Parameters
    ----------
    a : Array
    b : Array

    Returns
    -------
    out : Array
    """
    return a if b is None else b

@node
@required(1)
def first_colorarray(a,b):
    """
    First ColorArray

    Passess forward the first value it gets.

    Parameters
    ----------
    a : ColorArray
    b : ColorArray

    Returns
    -------
    out : ColorArray
    """
    return a if b is None else b

@node
def pass_int(cond,input):
    """
    Pass Int

    Passess forward the first value it gets.

    Parameters
    ----------
    cond : Int
    input : Int

    Returns
    -------
    out : Int
    """
    return input if cond else None

@node
def pass_float(cond,input):
    """
    Pass Float

    Passess forward the first value it gets.

    Parameters
    ----------
    cond : Int
    input : Float

    Returns
    -------
    out : Float
    """
    return input if cond else None

@node
def pass_array(cond,input):
    """
    Pass Array

    Passess forward the first value it gets.

    Parameters
    ----------
    cond : Int
    input : Array

    Returns
    -------
    out : Array
    """
    return input if cond else None

@node
def pass_color_array(cond,input):
    """
    Pass ColorArray

    Passess forward the first value it gets.

    Parameters
    ----------
    cond : Int
    input : ColorArray

    Returns
    -------
    out : ColorArray
    """
    return input if cond else None

@node
def pass_curve(cond,input):
    """
    Pass Curve

    Passess forward the first value it gets.

    Parameters
    ----------
    cond : Int
    input : Curve

    Returns
    -------
    out : Curve
    """
    return input if cond else None