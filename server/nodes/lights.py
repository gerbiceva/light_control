from datatypes import ColorArray, node, each_tick, initialize, generator
import jax.numpy as jnp
from matplotlib.colors import hsv_to_rgb
import sacn
import psutil
import socket

# def get_all_ips():
#     ips = []
#     try:
#         # Get all network interfaces
#         interfaces = netifaces.interfaces()
#         for interface in interfaces:
#             # Get all addresses for the interface
#             addresses = netifaces.ifaddresses(interface)
#             # Get IPv4 addresses
#             if netifaces.AF_INET in addresses:
#                 for link in addresses[netifaces.AF_INET]:
#                     ips.append(link['addr'])
#     except Exception as e:
#         return f"Error getting all IPs: {e}"
#     return ips


senders = {}
for interface, addrs in psutil.net_if_addrs().items():
    for addr in addrs:
        if addr.family == socket.AF_INET:
            senders[addr.address] = sacn.sACNsender(bind_address=addr.address, source_name="Best lighting software")

@generator
def make_lights():
    for ip, sender in senders.items():
        @node
        @initialize
        class Light:
            __doc__ = f"""
            Light Strip at {ip}

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

            def __init__(self):
                self.sender: sacn.sACNsender = sender
                self.sender.start()
                self.sender.manual_flush = True

            def __call__(self, universe, hsv: ColorArray):
                if universe not in self.sender.get_active_outputs():
                    self.sender.activate_output(universe)
                    self.sender[universe].multicast = True
                self.sender[universe].dmx_data = jnp.pad(
                    (hsv_to_rgb(jnp.clip(hsv, 0.0, 1.0).T).flatten() * 255).astype(jnp.uint8),
                    (0, 512 - hsv.shape[0] * hsv.shape[1]),
                    mode="constant",
                    constant_values=0,
                ).tobytes()

            def __del__(self):
                for uni in self.sender.get_active_outputs():
                    self.sender.deactivate_output(uni)
        yield Light

@each_tick
def send():
    for sender in senders.values():
        sender.flush()