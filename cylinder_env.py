import gymnasium as gym
from gymnasium import spaces
import numpy as np
from motor_controller import DCMotorController
from DAQ_class import DAQReader
from encoder_reader import EncoderReader
import time
# from encoder_reader import EncoderReader
from collections import deque
import datetime
from scipy.io import savemat


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action & Observation spaces
        self.action_space = spaces.Box(low=-0.4, high=0.4, shape=(1,), dtype=np.float32)
        self.history_length = 3 # can be any number
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(2 + self.history_length,),
                                            dtype=np.float32)
        
        # Hardware interfaces
        self.daq = DAQReader()
        self.motor = DCMotorController()
        self.encoder = EncoderReader()

        # System parameters
        self.bias_displacement = 470.8 
        self.bias_Fx = -0.25
        self.bias_Fy = -5.3066
        self.bias_Fz = 5.1341
        self.bias_acc = 1.5160
        self.mass = 1.095 # Kg mass of the system
        self.rho = 1000
        self.D = 0.0175 # Diameter of the cylinder in m
        self.L = 0.16 # Length of the cylinder in m
        self.U = 0.18
        self.factor = 0.5 * self.rho * self.U**2 * self.D * self.L
        self.tau = 2*3.14*1.96
        self.alpha_v = 0.2
        self.target_dt = 0.1 # 100 ms
        self.CPR = 5328 # counts per revolution 

        # State Variables
        self.state = np.zeros(2 + self.history_length, dtype=np.float32)
        self.step_count = 0
        self.prev_actions = deque([0.0] * self.history_length, maxlen=self.history_length)
        self.y_prev = None
        self.ydot_filtered = 0.0

        # Logging 
        self.action_log = []
        self.y_log = []
        self.ydot_log = []
        self.yddot_log = []
        self.Fx_log = []
        self.Fy_log = [] 
        self.rpm_log = []
        self.time_log = []

       # Timing
        self.prev_counts = self.encoder.get_position()
        self.last_time = time.perf_counter()
        self.start_time = self.last_time

    # ---------------- Reset ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the environment to an initial state
        self.step_count = 0
        self.prev_actions = deque([0.0] * self.history_length, maxlen=self.history_length)
        self.state = np.array(2 + self.history_length, dtype=np.float32) 
        self.y_prev = None
        self.ydot_filtered = 0.0
        return self.state, {}

    # ---------------- Step ----------------
    def step(self, action):
        # ----- Clip and apply action -----
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_velocity = float(action[0])
        self.motor.set_velocity(action_velocity)        
        self.prev_actions.append(action_velocity)

        # Read Sensors
        readings_vector, Fx_raw, Fy_raw, Fz_raw = self.daq.read()
        counts = self.encoder.get_position()
        delta_counts = counts - self.prev_counts
        revs = delta_counts / self.CPR
        current_time = time.perf_counter()
        dt = current_time - self.last_time            
        rpm = (revs / dt) * 60.0 if dt > 0 else 0.0
        self.prev_counts = counts
        self.last_time = current_time
        elapsed_time = current_time - self.start_time

        # ----- Process measurements -----
        raw_disp = ((75.0*readings_vector[1] - self.bias_displacement)/1000.0)
        raw_acc = ((readings_vector[0]) - self.bias_acc)/(0.3*9.81)
        raw_Fx = ((Fx_raw / 1000000.0) - self.bias_Fx)
        raw_Fy = (((Fy_raw / 1000000.0) - self.bias_Fy))

        # ----- Velocity filtering -----
        if self.y_prev is not None and dt > 0:
            velocity_raw = (raw_disp - self.y_prev) / dt
            velocity_filtered = self.alpha_v * velocity_raw + (1 - self.alpha_v) * self.ydot_filtered
            self.ydot_filtered = velocity_filtered
        else:
            velocity_filtered = 0.0
        self.y_prev = raw_disp

        # Nondimensionalization
        displacement = raw_disp / self.D
        velocity = velocity_filtered / (self.tau * self.D)
        acceleration = raw_acc / (self.tau**2 * self.D)
        Fx = raw_Fx / (self.factor)
        Fy = raw_Fy/(self.factor)

        # ----- Update state -----
        self.state = np.array([displacement, velocity] + list(self.prev_actions), dtype=np.float32)
        self.step_count += 1

        # ----- reward function -----
        reward = -(abs(displacement)) 

        # ----- Termination -----
        terminated = abs(displacement) > 5.0  # safety
        truncated = self.step_count >= 128  # max steps

        # ----- Maintain consistent step timing -----
        if dt < self.target_dt:
            time.sleep(self.target_dt - dt)

        self.action_log.append(action_velocity)
        self.y_log.append(displacement)
        self.ydot_log.append(velocity) 
        self.yddot_log.append(acceleration)
        self.Fx_log.append(Fx)
        self.Fy_log.append(Fy)
        self.rpm_log.append(rpm)
        self.time_log.append(elapsed_time)
        info = {}


        return self.state, reward, terminated, truncated, info

    # ---------------- Render ----------------
    def render(self, mode='human'):
        print(f"State: {self.state}")

    # ---------------- Save Logs ----------------
    def save_traininglogs_to_mat(self, filename="Training_logs.mat"):
        data = {
            "actions": np.array(self.action_log),
            "displacement": np.array(self.y_log),
            "velocity": np.array(self.ydot_log),
            "acceleration": np.array(self.yddot_log),
            "Fx": np.array(self.Fx_log),
            "Fy": np.array(self.Fy_log),
            "rpm": np.array(self.rpm_log),
            "time": np.array(self.time_log)
        }
        savemat(filename, data)
        print(f"[INFO] Saved {len(self.action_log)} steps to {filename}")

    # ---------------- Close Hardware ----------------
    def close(self):
        try:
            self.daq.close_ethercat()
            self.daq.close()
            self.motor.stop()
            self.motor.close()
            print("[Hardware Closed]")
        except Exception as e:
            print(f"[CLOSE ERROR] {e}")
