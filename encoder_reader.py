from Phidget22.Phidget import *
from Phidget22.Devices.Encoder import *

class EncoderReader:
    def __init__(self, serial_number=None):
        self.encoder = Encoder()

        # Optional: Set specific serial number if needed
        if serial_number is not None:
            self.encoder.setDeviceSerialNumber(serial_number)

        self.encoder.openWaitForAttachment(5000)
        self.encoder.setDataInterval(1)

        print("Encoder attached")

    def get_position(self):
        """Return current encoder position as integer"""
        return self.encoder.getPosition()

    def close(self):
        self.encoder.close()
        print("Encoder closed")