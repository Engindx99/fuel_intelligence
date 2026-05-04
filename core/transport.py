# transport.py - Kiln Simulation Component
import numpy as np

class TransportModel:
    def __init__(self, config):
        self.L = config['kiln']['length']
        self.D = config['kiln']['diameter']
        self.S = config['kiln']['slope']
        self.rpm = config['kiln']['rpm']

    def calculate_solid_velocity(self):
        """
        Sullivan-Friedman Denklemi: Katı faz eksenel hızı (m/s)
        """
        # v = (1.77 * D * rpm * S) / (sin(theta)) - basitleştirilmiş form
        v_s = (0.19 * self.D * self.rpm * self.S) / 60.0 # m/s
        return v_s

    def get_residence_time(self):
        return self.L / self.calculate_solid_velocity()