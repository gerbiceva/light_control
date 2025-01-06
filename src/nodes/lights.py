from datatypes import ColorArray
import jax.numpy as jnp
from matplotlib.colors import hsv_to_rgb
import sacn

sender = sacn.sACNsender(bind_address="192.168.0.103")
sender.start()
sender.manual_flush = True
for i in range(1, 6):
    sender.activate_output(i)
    sender[i].multicast = True

def light_strip(universe: int, hsv: ColorArray):
    # print(jnp.pad(
    #     (hsv_to_rgb(hsv.T).flatten() * 255).astype(jnp.uint8),
    #     (0, 512 - hsv.shape[0] * hsv.shape[1]),
    #     mode="constant",
    #     constant_values=0,
    # ))
    sender[universe].dmx_data = jnp.pad(
        (hsv_to_rgb(hsv.T).flatten() * 255).astype(jnp.uint8),
        (0, 512 - hsv.shape[0] * hsv.shape[1]),
        mode="constant",
        constant_values=0,
    ).tobytes()