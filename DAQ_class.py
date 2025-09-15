import nidaqmx
import numpy as np
from nidaqmx.constants import TerminalConfiguration, AcquisitionType, READ_ALL_AVAILABLE
import pysoem
import time
import ctypes

class DAQReader:
    def __init__(self, device_name="Dev1", sample_rate=1000.0):
        self.num_channels = 1
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.task = nidaqmx.Task()
        self.master = pysoem.Master()
        self._setup_daq_channels()
        self._setup_ethercat()

        self.task.start()

    def _setup_daq_channels(self):
        self.task.ai_channels.add_ai_voltage_chan(
        f"{self.device_name}/ai11",
        terminal_config=TerminalConfiguration.RSE,
        min_val=-10.0, max_val=10.0)          

        self.task.ai_channels.add_ai_voltage_chan(
        f"{self.device_name}/ai1",
        terminal_config=TerminalConfiguration.RSE,
        min_val=-10.0, max_val=10.0)        

    def _setup_ethercat(self):
        try:
            self.master.open('\\Device\\NPF_{12FB5FC4-5E55-45A6-8CAD-4CF392039EFF}') # device name
            if self.master.config_init() > 0:
                self.device = self.master.slaves[0]
                print(f'Found Device: {self.device.name}')
                self.master.config_map()        # <-- map PDOs
                self.master.config_dc()         # <-- optional: distributed clock sync
                self.master.state_check(pysoem.OP_STATE, 5_000_000)
            else:
                raise RuntimeError("No EtherCAT slaves found.")
        except Exception as e:
            print(f"EtherCAT setup failed: {e}")
            self.device = None

    def read(self):
        readings = self.task.read(number_of_samples_per_channel=1)
        readings = np.array(readings).flatten().astype(np.float32)
        if self.device is None:
            raise RuntimeError("EtherCAT device is not initialized.")

        try:
            # Exchange process data
            self.master.send_processdata()
            self.master.receive_processdata(1000) # 1000 time out in microseconds
            input_data = self.device.input  # raw PDO input buffer (typically 12 bytes)
            if len(input_data) < 12:
                Fx_raw = Fy_raw = Fz_raw = 0

            Fx_raw = ctypes.c_int32.from_buffer_copy(input_data[0:4]).value
            Fy_raw = ctypes.c_int32.from_buffer_copy(input_data[4:8]).value
            Fz_raw = ctypes.c_int32.from_buffer_copy(input_data[8:12]).value

        except pysoem.WkcError as e:
            print("[WKC Error] Failed to read from EtherCAT device.")
            Fx_raw = Fy_raw = Fz_raw = 0
        except Exception as e:
            print(f"[General EtherCAT Error] {e}")
            Fx_raw = Fy_raw = Fz_raw = 0

        return readings, Fx_raw, Fy_raw, Fz_raw  
    
    def close(self):
        if hasattr(self, 'task'):
            try:
                self.task.close()
            except Exception as e:
                print(f"[DAQ close error] {e}")

    def __del__(self):
        self.close()
        self.close_ethercat()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self.close_ethercat()


    def close_ethercat(self):
        if self.master:
            try:
                # Optional: Set the slave device to INIT state before closing
                for slave in self.master.slaves:
                    slave.state = pysoem.INIT_STATE
                    slave.write_state()

                self.master.state_check(pysoem.INIT_STATE, 5_000_000)

                self.master.close()
                print("EtherCAT connection closed.")
            except Exception as e:
                print(f"[EtherCAT close error] {e}")
