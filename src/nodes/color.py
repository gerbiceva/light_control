from datatypes import ColorArray, Color, Array

def set_brightness_all(hsv: ColorArray, value: float) -> ColorArray:
    return hsv.at[2, :].set(value)

def set_saturation_all(hsv: ColorArray, value: float) -> ColorArray:
    return hsv.at[1, :].set(value)

def set_hue_all(hsv: ColorArray, value: float) -> ColorArray:
    return hsv.at[0, :].set(value)

def set_color_all(hsv: ColorArray, color: Color) -> ColorArray:
    return hsv.at[:, :].set(color)

def set_hue_array(hsv: ColorArray, array: Array) -> ColorArray:
    return hsv.at[0, :].set(array)

def set_saturation_array(hsv: ColorArray, array: Array) -> ColorArray:
    return hsv.at[1, :].set(array)

def set_brightness_array(hsv: ColorArray, array: Array) -> ColorArray:
    return hsv.at[2, :].set(array)