from device_drivers import Gamepad, Microphone
from datatypes import Vector2D
import jax.numpy as jnp

buttons = [
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
pad = Gamepad("/dev/input/event17")
mic = Microphone(3)

def get_gamepad_state() -> (int, int, int, int, int, int, int, int, int, int, int, int, int, int, float, float, Vector2D, Vector2D):
    return tuple(pad.buttons[button] for button in buttons) + (
        jnp.array(pad.buttons["R2"]),
        jnp.array(pad.buttons["L2"]),
        jnp.array(pad.sticks["Left"]),
        jnp.array(pad.sticks["Right"]),
    )

def get_gain() -> float:
    return mic.gain
