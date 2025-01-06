from evdev import InputDevice, ecodes
import threading
import jax.numpy as jnp
import numpy as np
import pyaudio
from utils import FrameLimiter

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
            "Left": jnp.array([0.0, 0.0]),
            "Right": jnp.array([0.0, 0.0])
        }
        self.thread.start()

    def loop(self):
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
                    self.buttons["L2"] = event.value / 255
                elif event.code == ecodes.ABS_RZ:
                    self.buttons["R2"] = event.value / 255

class Microphone:
    def __init__(self, input_device_index, fps=30):
        self.input_device_index = input_device_index
        self.fps = fps
        self.gain = 0.0
        self.running = True
        self.thread = threading.Thread(target=self.loop)

        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Configure the audio stream
        self._setup_stream()

        # Start the processing thread
        self.thread.start()

    def _setup_stream(self):
        """Initialize the audio stream."""
        device_info = self.audio.get_device_info_by_index(self.input_device_index)
        max_input_channels = device_info["maxInputChannels"]

        if max_input_channels == 0:
            raise ValueError(f"Device {self.input_device_index} does not support input.")

        self.stream = self.audio.open(
            format=pyaudio.paInt16,  # 16-bit audio format
            channels=1,             # Mono audio
            rate=44100,             # Sample rate
            input=True,             # Capture audio from microphone
            frames_per_buffer=1470, # Adjust buffer size for 30 FPS
            input_device_index=self.input_device_index
        )

    def calculate_gain(self, data):
        """Calculate RMS gain using JAX, with handling for NaN or Inf values."""
        audio_array = jnp.array(np.frombuffer(data, dtype=np.int16))
        audio_array = jnp.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
        rms = jnp.sqrt(jnp.mean(audio_array**2))
        return float(rms) if not (jnp.isnan(rms) or jnp.isinf(rms)) else 0.0

    def loop(self):
        """Continuously read audio and calculate gain."""
        frame_limiter = FrameLimiter(self.fps)

        while self.running:
            try:
                # Read audio data
                data = self.stream.read(1470, exception_on_overflow=False)
                self.gain = self.calculate_gain(data)

                # Limit processing to the specified FPS
                frame_limiter.tick()
            except Exception as e:
                print(f"Error in audio loop: {e}")
                self.running = False

    def stop(self):
        """Stop the microphone processing."""
        self.running = False
        self.thread.join()

        # Clean up audio resources
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def __del__(self):
        """Ensure resources are cleaned up on object deletion."""
        self.stop()


if __name__=="__main__":
    pad = Gamepad()
