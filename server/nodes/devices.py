from datatypes import Vector2D, node, initialize, Float, Int, generator
import jax.numpy as jnp
from evdev import InputDevice, ecodes, list_devices
import threading

@generator
def make_gamepads():    
    devices: list[InputDevice] = []
    for device in [InputDevice(path) for path in list_devices()]:
        if "Controller" in device.name or "Wireless" in device.name:
            devices.append(InputDevice(device.path))

    gamepads = []
    for device in devices:
        @node
        @initialize
        class Gamepad:
            __doc__ = f"""
            Gamepad Input {device.name}

            Returns all buttons triggers and sticks. Buttons are integers 0 or 1.
            Triggers are floats from 0-1. Sticks are Vector2Ds defined from -1 to 1.

            Returns
            -------
            A : Int
            B : Int
            X : Int
            Y : Int
            L1 : Int
            R1 : Int
            Start : Int
            Select : Int
            L3 : Int
            R3 : Int
            Left : Int
            Right : Int
            Up : Int
            Down : Int
            R2 : Float
            L2 : Float
            LeftStick : Vector2D
            RightStick : Vector2D
            """
            button_names = [
                    "A",
                    "B",
                    "X",
                    "Y",
                    "L1",
                    "R1",
                    "Start",
                    "Select",
                    "L3",
                    "R3",
                    "Left",
                    "Right",
                    "Up",
                    "Down",
                ]
            def __init__(self):
                self.thread = threading.Thread(target=self.loop)
                self.device = device
                self.buttons = {
                    "A": 0,
                    "B": 0,
                    "X": 0,
                    "Y": 0,
                    "L1": 0,
                    "R1": 0,
                    "L2": 0.0,
                    "R2": 0.0,
                    "Start": 0,
                    "Select": 0,
                    "L3": 0,
                    "R3": 0,
                    "Left": 0,
                    "Right": 0,
                    "Up": 0,
                    "Down": 0
                }
                self.codes_to_buttons = {
                    ecodes.BTN_SOUTH: "A",
                    ecodes.BTN_EAST: "B",
                    ecodes.BTN_NORTH: "X",
                    ecodes.BTN_WEST: "Y",
                    ecodes.BTN_TL: "L1",
                    ecodes.BTN_TR: "R1",
                    ecodes.BTN_TL2: "L2",
                    ecodes.BTN_TR2: "R2",
                    ecodes.BTN_START: "Start",
                    ecodes.BTN_SELECT: "Select",
                    ecodes.BTN_THUMBL: "L3",
                    ecodes.BTN_THUMBR: "R3",
                }

                self.sticks = {
                    "Left": jnp.array([0.0, 0.0]),
                    "Right": jnp.array([0.0, 0.0])
                }
                self.stop = False
                self.thread.start()

            def loop(self):
                if self.stop:
                    return
                for event in self.device.read_loop():
                    if event.type == ecodes.EV_KEY:  # Button presses
                        self.buttons[self.codes_to_buttons[event.code]] = 1 if event.value else 0
                    elif event.type == ecodes.EV_ABS:  # Analog inputs (axes)
                        if event.code == ecodes.ABS_X:
                            self.sticks["Left"] = self.sticks["Left"].at[0].set(event.value / 32768.0)
                        elif event.code == ecodes.ABS_Y:
                            self.sticks["Left"] = self.sticks["Left"].at[1].set(event.value / 32768.0)
                        elif event.code == ecodes.ABS_RX:
                            self.sticks["Right"] = self.sticks["Right"].at[0].set(event.value / 32768.0)
                        elif event.code == ecodes.ABS_RY:
                            self.sticks["Right"] = self.sticks["Right"].at[1].set(event.value / 32768.0)
                        elif event.code == ecodes.ABS_HAT0X:
                            if event.value == -1:
                                self.buttons["Left"] = 1
                            elif event.value == 1:
                                self.buttons["Right"] = 1
                            else:
                                self.buttons["Left"] = 0
                                self.buttons["Right"] = 0
                        elif event.code == ecodes.ABS_HAT0Y:
                            if event.value == -1:
                                self.buttons["Up"] = 1
                            elif event.value == 1:
                                self.buttons["Down"] = 1
                            else:
                                self.buttons["Up"] = 0
                                self.buttons["Down"] = 0
                        elif event.code == ecodes.ABS_Z:
                            # print(event)
                            self.buttons["L2"] = float(event.value) / 255.0
                        elif event.code == ecodes.ABS_RZ:
                            self.buttons["R2"] = float(event.value) / 255.0

            def __call__(self) -> (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Float, Float, Vector2D, Vector2D):
                return tuple(self.buttons[button] for button in self.button_names) + (
                    jnp.array(self.buttons["R2"]),
                    jnp.array(self.buttons["L2"]),
                    jnp.array(self.sticks["Left"]),
                    jnp.array(self.sticks["Right"]),
                )

            def __del__(self):
                self.stop = True
                # self.thread.join()
        gamepads.append(Gamepad)
    return gamepads