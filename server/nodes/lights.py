from datatypes import ColorArray, node, each_tick
import jax.numpy as jnp
from matplotlib.colors import hsv_to_rgb
import sacn

sender = sacn.sACNsender(bind_address="0.0.0.0", source_name="Best lighting software")
sender.start()
sender.manual_flush = True

for i in range(1, 10):
    sender.activate_output(i)
    sender[i].multicast = True


@each_tick
def send():
    sender.flush()


@node
def light_strip(universe: int, hsv: ColorArray):
    """
    Control Light Strip with HSV Color Array

    Sends HSV color data to a specified DMX universe, converting it to RGB format
    and padding the data to meet the DMX512 protocol requirements.

    Parameters
    ----------
    universe : Int
        The DMX universe to which the color data will be sent.
    hsv : ColorArray
        A color array in HSV format, where each color is represented as a vector.

    Returns
    -------
    None
    """
    # print(hsv[:, 1])
    sender[universe].dmx_data = jnp.pad(
        (hsv_to_rgb(jnp.clip(hsv, 0.0, 1.0).T).flatten() * 255).astype(jnp.uint8),
        (0, 512 - hsv.shape[0] * hsv.shape[1]),
        mode="constant",
        constant_values=0,
    ).tobytes()
