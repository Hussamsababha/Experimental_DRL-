from Phidget22.Phidget import *
from Phidget22.Devices.DCMotor import *
import time

class DCMotorController:
    def __init__(self, serial_number=None):
        self.motor = DCMotor()

        # Optional: Set addressing parameters (e.g., serial number, hub port)
        if serial_number is not None:
            self.motor.setDeviceSerialNumber(serial_number)

        self.motor.openWaitForAttachment(5000)
        print("DC Motor attached")
        self.motor.setDataInterval(100) # Set data interval to 100 ms, the minimum

    def set_velocity(self, velocity: float):
        """Set target velocity between -1.0 and 1.0"""
        velocity = max(-1.0, min(1.0, velocity))  # Clamp value to range
        self.motor.setTargetVelocity(velocity)

    def stop(self):
        """Stop the motor safely"""
        self.motor.setTargetVelocity(0.0)
        print("Motor stopped")

    def close(self):
        self.motor.close()
        print("Motor closed")