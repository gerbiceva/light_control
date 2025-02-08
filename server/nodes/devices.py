from gamepad import Gamepad
from datatypes import Vector2D, node, initialize, Float
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
# pad = Gamepad("/dev/input/event17")
# mic = Microphone(3)
@node
@initialize
def make_gamepad():
    """
    Gamepad Input

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
    pad = Gamepad()
    def get_gamepad_state() -> (int, int, int, int, int, int, int, int, int, int, int, int, int, int, float, float, Vector2D, Vector2D):
        return tuple(pad.buttons[button] for button in buttons) + (
            jnp.array(pad.buttons["R2"]),
            jnp.array(pad.buttons["L2"]),
            jnp.array(pad.sticks["Left"]),
            jnp.array(pad.sticks["Right"]),
        )
    return get_gamepad_state