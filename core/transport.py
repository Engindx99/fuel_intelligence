import numpy as np

class TransportModel:
    def __init__(self, config):
        # Statik geometrik parametreler
        self.cfg = config
        self.L = config['kiln']['length']
        self.D = config['kiln']['diameter']
        self.S = config['kiln']['slope']

    def calculate_solid_velocity(self):
        """
        Sullivan-Friedman Denklemi: Katı faz eksenel hızı (m/s)
        Dinamik RPM okuması yapar.
        """
        # Güncel RPM değerini çek (Kontrolcü tarafından değiştirilmiş olabilir)
        current_rpm = self.cfg['kiln']['rpm']
        if isinstance(current_rpm, list): current_rpm = current_rpm[0]
        
        # v_s (m/s) hesabı
        v_s = (0.19 * self.D * float(current_rpm) * self.S) / 60.0 
        return max(1e-4, v_s) # Bölme hatası için alt limit

    def get_residence_time(self):
        return self.L / self.calculate_solid_velocity()