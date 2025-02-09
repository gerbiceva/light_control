import gi
import threading
from datatypes import node, initialize, thread
import weakref
import jax.numpy as jnp

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

@node
@initialize
class MicGain:
    """
    Microphone Gain

    Returns gain from mic.

    Returns
    -------
    gain : Float
    """

    def __init__(self):
        if not hasattr(self, "fps"):
            self.fps = 60

        # Create a pipeline with a level element to measure audio levels
        pipeline_description = (
            f"autoaudiosrc ! audioconvert ! level interval={int((1 / self.fps) * 10**9)} ! fakesink"
        )
        self.pipeline = Gst.parse_launch(pipeline_description)

        # Start the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.peak = -10.0
        weak_self_get = weakref.ref(self)

        # Function to handle messages from the pipeline
        def on_message(bus, message):
            weak_self = weak_self_get()
            if message.type == Gst.MessageType.ELEMENT and weak_self is not None:
                structure = message.get_structure()
                if structure and structure.get_name() == "level":
                    weak_self.peak = structure.get_value("peak")[0]

            # Return True to keep the bus watcher running
            return True

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_message)

    def __call__(self):
        return self.peak

    def __del__(self):
        self.pipeline.set_state(Gst.State.NULL)


@node
@initialize
class MicFourier:
    """
    Microphone Spectrum

    Returns frequency spectrum from mic.

    Returns
    -------
    spectrum : Array
    """

    def __init__(self):
        if not hasattr(self, "fps"):
            self.fps = 60

        # Create a pipeline with a level element to measure audio levels
        pipeline_description = pipeline_description = (
            "autoaudiosrc ! "
            "audioconvert ! "
            f"spectrum bands=128 interval={int((1 / self.fps) * 10**9)} ! "
            "fakesink"
        )
        self.pipeline = Gst.parse_launch(pipeline_description)

        self.pipeline.set_state(Gst.State.PLAYING)
        weak_self_get = weakref.ref(self)
        self.magnitude = None

        # Function to handle messages from the pipeline
        def on_message(bus, message):
            weak_self = weak_self_get()
            if message.type == Gst.MessageType.ELEMENT and weak_self is not None:
                structure = message.get_structure()
                if structure and structure.get_name() == "spectrum":
                    weak_self.magnitude = (structure.get_value("magnitude"))

            # Return True to keep the bus watcher running
            return True

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_message)

    def __call__(self):
        return jnp.array(self.magnitude)

    def __del__(self):
        self.pipeline.set_state(Gst.State.NULL)


@thread
class MainLoopThread:
    def __init__(self):
        self.loop = GLib.MainLoop()
        threading.Thread(target=self.loop.run).start()

    def stop(self):
        self.loop.quit()
