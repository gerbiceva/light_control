from datatypes import ColorArray, node, each_tick, initialize, generator
import jax.numpy as jnp
from matplotlib.colors import hsv_to_rgb
import sacn
import netifaces


senders = {}

def get_ipv4_gateway_ips():
    gateway_ips = {}

    # Get all gateways
    gateways = netifaces.gateways()

    # Extract only IPv4 gateways
    ipv4_gateways = gateways.get(netifaces.AF_INET, [])

    for gateway in ipv4_gateways:
        interface = gateway[1]  # Interface name
        gateway_ip = gateway[0]  # Gateway IP address
        gateway_ips[interface] = gateway_ip

    # return gateway_ips
    return {'buu': '0.0.0.0'}

@generator
def make_lights():    
    for addr in get_ipv4_gateway_ips().values():
        senders[addr] = sacn.sACNsender(bind_address=addr, source_name="Best lighting software")
        senders[addr].start()
        senders[addr].manual_flush = True
    lights = []
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
                self.sender.is_ready = True

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
        lights.append(Light)
    return lights

@each_tick
def send():
    for sender in senders.values():
        if hasattr(sender, 'is_ready'):
            # print(sender.get_active_outputs())
            sender.flush()