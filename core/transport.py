import numpy as np


class TransportModel:
    def __init__(self, config):

        kiln_cfg = config.get("kiln", {})

        self.L = float(kiln_cfg.get("length", 60.0))
        self.D = float(kiln_cfg.get("diameter", 4.2))
        self.S = float(kiln_cfg.get("slope", 0.03))

        # fiziksel sınırlar
        self.fill_min = 0.05
        self.fill_max = 0.18

    def get_dynamic_filling_degree(self, current_rpm, feed_rate_ton_h=None):

        rpm = float(current_rpm)
        rpm = max(rpm, 0.5)

        base_fill = 0.10

        # RPM etkisi (stabilize edildi)
        rpm_factor = (2.0 / rpm) ** 0.15
        dynamic_fill = base_fill * rpm_factor

        if feed_rate_ton_h is not None:

            feed_rate_ton_h = float(feed_rate_ton_h)

            # daha yumuşak scaling (overreaction fix)
            feed_factor = (feed_rate_ton_h / 125.0) ** 0.10
            dynamic_fill *= feed_factor

        return np.clip(dynamic_fill, self.fill_min, self.fill_max)

    def calculate_solid_velocity(self, current_rpm, fill_degree=None):

        rpm = float(current_rpm)
        rpm = max(rpm, 0.1)

        # temel axial velocity
        v_base = (1.77 * rpm * self.D * self.S) / 60.0

        v_s = v_base

        if fill_degree is not None:

            fd = float(fill_degree)
            fd = max(fd, 0.05)

            # stabilize edilmiş correction (önceki model çok agresifti)
            fill_correction = (0.10 / fd) ** 0.15
            v_s *= fill_correction

        # 🔥 CRITICAL FIX: fiziksel üst limit (residence time koruma)
        v_s = min(v_s, 0.12)

        return max(1e-4, v_s)

    def check_cfl_condition(self, v_s, dt, nodes):

        dz = self.L / nodes
        max_dt = dz / max(v_s, 1e-12)

        # güvenlik margin
        safe_dt = 0.8 * max_dt

        # 🔥 CRITICAL FIX: dt sadece “uyarı” değil, enforce edilmeli
        if dt > safe_dt:
            dt = safe_dt
            return False, dt

        return True, dt