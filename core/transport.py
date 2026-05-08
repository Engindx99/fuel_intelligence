import numpy as np


class TransportModel:
    def __init__(self, config):
        self.cfg = config
        self.L = float(config['kiln']['length'])
        self.D = float(config['kiln']['diameter'])
        self.S = float(config['kiln']['slope'])

        # Nominal filling sabit tutulur (physics separation)
        self.base_fill = float(config['kiln'].get('filling_degree', 0.10))

    def calculate_solid_velocity(self, current_rpm=None):

        if current_rpm is None:
            current_rpm = float(self.cfg['kiln']['rpm'])

        # =====================================================
        # PHYSICALLY STABLE SCALING
        # v ~ sqrt(RPM) yaklaşımı (granular flow consistent)
        # =====================================================

        rpm_effective = np.sqrt(max(1e-6, current_rpm))

        v_s = (
            0.19 *
            self.D *
            rpm_effective *
            self.S
        ) / 60.0

        return max(1e-4, v_s)

    def get_dynamic_filling_degree(self, current_rpm):
        """
        RPM artık doğrudan geometry modulator değil.
        Sadece zayıf correction olarak tutulur.
        """

        base_fill = self.base_fill

        # Çok zayıf coupling (instability suppression)
        rpm_factor = (2.0 / max(0.5, current_rpm))**0.08

        dynamic_fill = base_fill * rpm_factor

        # fiziksel clamp
        return np.clip(dynamic_fill, 0.08, 0.12)