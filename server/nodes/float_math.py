from datatypes import node, Float, Int

@node
def float_to_int(a: Float) -> Int:
    """
    Float 2 Int

    Converts float to integer.

    Parameters
    ----------
    in : Float

    Returns
    -------
    out : Int
    """
    return int(a)

@node
def add(a: Float, b: Float) -> Float:
    """
    Add

    Adds two numbers.

    Parameters
    ----------
    a : Float
        The first number.
    b : Float
        The second number.

    Returns
    -------
    result : Float
    """
    return a + b


@node
def subtract(a: Float, b: Float) -> Float:
    """
    Subtract

    Subtracts the second number from the first.

    Parameters
    ----------
    a : Float
        The number to subtract from.
    b : Float
        The number to subtract.

    Returns
    -------
    result : Float
    """
    return a - b


@node
def multiply(a: Float, b: Float) -> Float:
    """
    Multiply

    Multiplies two numbers.

    Parameters
    ----------
    a : Float
        The first number.
    b : Float
        The second number.

    Returns
    -------
    result : Float
    """
    return a * b


@node
def divide(a: Float, b: Float) -> Float:
    """
    Divide
    
    Divides the dividend by the divisor.

    Parameters
    ----------
    a : Float
        The dividend of the division.
    b : Float
        The divisor of the division.

    Returns
    -------
    result : Float
    """
    # print(a)
    # print(b)
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

@node
def mod(a: Float, b: Float) -> Float:
    """
    Modulus

    Computes the remainder of the division of a by b.

    Parameters
    ----------
    a : Float
        The dividend.
    b : Float
        The divisor.

    Returns
    -------
    result : Float
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a % b


@node
def clamp(value: Float, min_value: Float, max_value: Float) -> Float:
    """
    Clamp

    Clamps a value between a minimum and a maximum.

    Parameters
    ----------
    value : Float
        The value to clamp.
    min_value : Float
        The minimum value.
    max_value : Float
        The maximum value.

    Returns
    -------
    result : Float
    """
    return max(min_value, min(value, max_value))
