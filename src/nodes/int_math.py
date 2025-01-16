from datatypes import node, Int

@node
def add(a: Int, b: Int) -> Int:
    """
    Add

    Adds two numbers.

    Parameters
    ----------
    a : Int
        The first number.
    b : Int
        The second number.

    Returns
    -------
    result : Int
    """
    return a + b


@node
def subtract(a: Int, b: Int) -> Int:
    """
    Subtract

    Subtracts the second number from the first.

    Parameters
    ----------
    a : Int
        The number to subtract from.
    b : Int
        The number to subtract.

    Returns
    -------
    result : Int
    """
    return a - b


@node
def multiply(a: Int, b: Int) -> Int:
    """
    Multiply

    Multiplies two numbers.

    Parameters
    ----------
    a : Int
        The first number.
    b : Int
        The second number.

    Returns
    -------
    result : Int
    """
    return a * b


@node
def divide(a: Int, b: Int) -> Int:
    """
    Divide
    
    Divides the dividend by the divisor.

    Parameters
    ----------
    a : Int
        The dividend of the division.
    b : Int
        The divisor of the division.

    Returns
    -------
    result : Int
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a // b

@node
def mod(a: Int, b: Int) -> Int:
    """
    Modulus

    Computes the remainder of the division of a by b.

    Parameters
    ----------
    a : Int
        The dividend.
    b : Int
        The divisor.

    Returns
    -------
    result : Int
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a % b


@node
def clamp(value: Int, min_value: Int, max_value: Int) -> Int:
    """
    Clamp

    Clamps a value between a minimum and a maximum.

    Parameters
    ----------
    value : Int
        The value to clamp.
    min_value : Int
        The minimum value.
    max_value : Int
        The maximum value.

    Returns
    -------
    result : Int
    """
    return max(min_value, min(value, max_value))
