import numpy as np


class TransportModel:
    def __init__(self, config):

        kiln_cfg = config.get("kiln", {})

        self.L = float(kiln_cfg.get("length", 60.0))
        self.D = float(kiln_cfg.get("diameter", 4.2))
        self.S = float(kiln_cfg.get("slope", 0.03))

        # 🔧 FIX: fiziksel limitler
        self.fill_min = 0.05
        self.fill_max = 0.18

    def get_dynamic_filling_degree(self, current_rpm, feed_rate_ton_h=None):

        rpm = float(current_rpm)
        if rpm < 0.5:
            rpm = 0.5

        base_fill = 0.10

        # RPM etkisi (daha yumuşak)
        rpm_factor = (2.0 / rpm) ** 0.30
        dynamic_fill = base_fill * rpm_factor

        if feed_rate_ton_h is not None:
            # feed etkisi daha fiziksel sınırlandırıldı
            feed_factor = (float(feed_rate_ton_h) / 125.0) ** 0.15
            dynamic_fill *= feed_factor

        return np.clip(dynamic_fill, self.fill_min, self.fill_max)

    def calculate_solid_velocity(self, current_rpm, fill_degree=None):

        rpm = float(current_rpm)
        if rpm < 0.1:
            rpm = 0.1

        v_base = (1.77 * rpm * self.D * self.S) / 60.0

        if fill_degree is not None:

            fd = float(fill_degree)
            if fd < 0.05:
                fd = 0.05

            # daha stabil correction
            fill_correction = (0.10 / fd) ** 0.35
            v_s = v_base * fill_correction
        else:
            v_s = v_base

        return max(1e-3, v_s)

    def check_cfl_condition(self, v_s, dt, nodes):

        dz = self.L / nodes
        max_dt = dz / max(v_s, 1e-9)

        # 🔧 FIX: margin ekledim (sayısal güvenlik)
        safe_dt = 0.8 * max_dt

        if dt > safe_dt:
            return False, safe_dt

        return True, safe_dt