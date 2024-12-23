class Light:
    def __init__(self, universe, sender, nm_leds):
        self.universe = universe
        sender.activate_output(universe)
        sender[universe].multicast = True

    @colors.setter
    def colors(self, hsv):
        sender[7].dmx_data = 
            np.pad((hsv_to_rgb(strip.T).flatten() * 255)
                       .astype(np.uint8),
                    (0, 212), mode='constant',
                    constant_values=0).tobytes()
