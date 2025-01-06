from device_drivers import Gamepad, Microphone

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
pad = Gamepad("/dev/input/event8")
mic = Microphone(3)

def get_gamepad_state() -> (int, int, int, int, int, int, int, int, int, int, int, int, int, int, float, float, list[float, float], list[float, float]):
    return tuple(pad.buttons[button] for button in buttons) + (
        pad.buttons["R2"],
        pad.buttons["L2"],
        pad.sticks["Left"],
        pad.sticks["Right"],
    )

def get_gain() -> float:
    return mic.gain
