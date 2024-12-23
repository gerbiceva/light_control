import time
import numpy as np

class FrameLimiter:
    def __init__(self, fps):
        self.interval = 1.0 / fps  # Time per frame in seconds
        self.last_time = time.time()

    def set_fps(fps):
        self.interval = 1.0 / fps

    def tick(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        sleep_time = max(0, self.interval - elapsed)
        time.sleep(sleep_time)
        self.last_time = time.time()  # Reset for the next frame

def print_strip(strip):
    bar = ""
    strip = (hsv_to_rgb(strip.T).flatten() * 255
    for i in range(strip.shape[1]):
        r, g, b = strip[:, i]
        bar += f"\033[48;2;{r};{g};{b}m  \033[0m"
    print(f"\r{bar}", end="")
