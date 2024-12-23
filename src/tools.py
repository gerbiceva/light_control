from evdev import InputDevice, categorize, ecodes, list_devices
devices = [InputDevice(path) for path in list_devices()]
for device in devices:
    print(f"Device: {device.name} at {device.path}")
