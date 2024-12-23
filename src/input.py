from evdev import InputDevice, categorize, ecodes
import threading

# List all input devices
# from evdev import list_devices
# devices = [InputDevice(path) for path in list_devices()]
# for device in devices:
#     print(f"Device: {device.name} at {device.path}")

class Gamepad():
    def __init__(self, input_device):
        self.device = InputDevice(input_device)
        self.thread = threading.Thread(target=self.loop)
        self.buttons = {
            "A": 0,
            "B": 0,
            "X": 0,
            "Y": 0,
            "L1": 0,
            "R1": 0,
            "L2": 0,
            "R2": 0,
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
            "Left": [0, 0],
            "Right": [0, 0]
        }
        self.thread.start()

    def loop(self):
        for event in self.device.read_loop():
            if event.type == ecodes.EV_KEY:  # Button presses
                self.buttons[self.codes_to_buttons[event.code]] = 1 if event.value else 0
            elif event.type == ecodes.EV_ABS:  # Analog inputs (axes)
                if event.code == ecodes.ABS_X:
                    self.sticks["Left"][0] = event.value / 32768
                elif event.code == ecodes.ABS_Y:
                    self.sticks["Left"][1] = event.value / 32768
                elif event.code == ecodes.ABS_RX:
                    self.sticks["Right"][0] = event.value / 32768
                elif event.code == ecodes.ABS_RY:
                    self.sticks["Right"][1] = event.value / 32768
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
                    self.buttons["L2"] = event.value / 255
                elif event.code == ecodes.ABS_RZ:
                    self.buttons["R2"] = event.value / 255


if __name__=="__main__":
    pad = Gamepad()
