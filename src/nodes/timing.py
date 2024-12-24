from ..node import Node
import time
from ..data import Float, Integer

class TimeNode(Node):
    def __init__(self):
        super().__init__(self.get_time)
        self.outputs["Seconds (Float)"] = Float()
        self.outputs["Seconds (Integer)"] = Integer()

    def get_time(self):
        pass