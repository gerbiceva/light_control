from datatypes import ColorArray, Color, Array, node

@node
def set_brightness_all(hsv: ColorArray, value: float) -> ColorArray:
    """
    Set Brightness of ColorArray

    Sets brightness of all colors in a color array 

    Parameters
    ----------
    hsv : ColorArray
        A color array in HSV format.
    value : Float
        The brightness value to set for all colors.

    Returns
    -------
    hsv : ColorArray
        The updated color array with modified brightness.
    """
    return hsv.at[2, :].set(value)


@node
def set_saturation_all(hsv: ColorArray, value: float) -> ColorArray:
    """
    Set Saturation of ColorArray

    Sets saturation of all colors in a color array 

    Parameters
    ----------
    hsv : ColorArray
        A color array in HSV format.
    value : Float
        The saturation value to set for all colors.

    Returns
    -------
    hsv : ColorArray
        The updated color array with modified saturation.
    """
    return hsv.at[1, :].set(value)

@node
def set_hue_all(hsv: ColorArray, value: float) -> ColorArray:
    """
    Set Hue of ColorArray

    Sets hue of all colors in a color array 

    Parameters
    ----------
    hsv : ColorArray
        A color array in HSV format.
    value : Float
        The hue value to set for all colors.

    Returns
    -------
    hsv : ColorArray
        The updated color array with modified hue.
    """
    return hsv.at[0, :].set(value)

@node
def set_color_all(hsv: ColorArray, color: Color) -> ColorArray:
    """
    Set Color for ColorArray

    Sets a specific color for all colors in a color array 

    Parameters
    ----------
    hsv : ColorArray
        A color array in HSV format.
    color : Color
        The color value to set for all colors.

    Returns
    -------
    hsv : ColorArray
        The updated color array with the new color.
    """
    return hsv.at[:, :].set(color)

@node
def set_hue_array(hsv: ColorArray, array: Array) -> ColorArray:
    """
    Set Hue using Array

    Sets hue values for a color array using a specific array of hue values 

    Parameters
    ----------
    hsv : ColorArray
        A color array in HSV format.
    array : Array
        An array of hue values to set.

    Returns
    -------
    hsv : ColorArray
        The updated color array with modified hue values.
    """
    return hsv.at[0, :].set(array)

@node
def set_saturation_array(hsv: ColorArray, array: Array) -> ColorArray:
    """
    Set Saturation using Array

    Sets saturation values for a color array using a specific array of saturation values 

    Parameters
    ----------
    hsv : ColorArray
        A color array in HSV format.
    array : Array
        An array of saturation values to set.

    Returns
    -------
    hsv : ColorArray
        The updated color array with modified saturation values.
    """
    return hsv.at[1, :].set(array)

@node
def set_brightness_array(hsv: ColorArray, array: Array) -> ColorArray:
    """
    Set Brightness using Array

    Sets brightness values for a color array using a specific array of brightness values 

    Parameters
    ----------
    hsv : ColorArray
        A color array in HSV format.
    array : Array
        An array of brightness values to set.

    Returns
    -------
    hsv : ColorArray
        The updated color array with modified brightness values.
    """
    return hsv.at[2, :].set(array)
