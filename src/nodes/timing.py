from node import Node
import time
from data import Float, Integer

class TimeNode(Node):
    def __init__(self):
        super().__init__(self.get_time)
        self.outputs["Seconds (Float)"] = Float(self)
        self.outputs["Seconds (Integer)"] = Integer(self)

    def get_time(self) -> (float, int):
        current = time.time()
        return current, int(current)