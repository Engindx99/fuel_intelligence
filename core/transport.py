import numpy as np


class TransportModel:

    def __init__(self, config):

        self.cfg = config

        self.L = float(
            self._safe_f(
                config["kiln"]["length"]
            )
        )

        self.D = float(
            self._safe_f(
                config["kiln"]["diameter"]
            )
        )

        self.S = float(
            self._safe_f(
                config["kiln"]["slope"]
            )
        )

    # ==============================================================
    # SAFE FLOAT
    # ==============================================================

    def _safe_f(self, value):

        if isinstance(value, list):
            return float(value[0])

        return float(value)

    # ==============================================================
    # SOLID VELOCITY
    # ==============================================================

    def calculate_solid_velocity(
        self,
        current_rpm=None,
        fill_degree=0.12
    ):
        """
        Rotary kiln axial solid transport velocity.

        Sullivan-type correlation with filling correction.
        """

        if current_rpm is None:

            current_rpm = float(
                self._safe_f(
                    self.cfg["kiln"]["rpm"]
                )
            )

        rpm = max(0.1, current_rpm)

        # Sullivan correlation
        v_base = (
            1.77
            * rpm
            * self.D
            * self.S
        ) / 60.0

        # Deeper bed -> slower transport
        fill_correction = (
            0.12
            / max(fill_degree, 0.05)
        ) ** 0.5

        v_s = (
            v_base
            * fill_correction
        )

        return max(0.002, v_s)

    # ==============================================================
    # FILLING DEGREE
    # ==============================================================

    def get_dynamic_filling_degree(
        self,
        current_rpm,
        feed_rate=None
    ):
        """
        Approximate kiln filling fraction.
        """

        if current_rpm is None:

            current_rpm = float(
                self._safe_f(
                    self.cfg["kiln"]["rpm"]
                )
            )

        rpm = max(0.5, current_rpm)

        base_fill = 0.12

        dynamic_fill = (
            base_fill
            * (3.5 / rpm) ** 0.35
        )

        # Optional throughput correction
        if feed_rate is not None:

            feed_factor = (
                feed_rate / 125.0
            ) ** 0.15

            dynamic_fill *= feed_factor

        return np.clip(
            dynamic_fill,
            0.06,
            0.18
        )

    # ==============================================================
    # RESIDENCE TIME
    # ==============================================================

    def calculate_residence_time(
        self,
        current_rpm,
        fill_degree=0.12
    ):
        """
        Mean solids residence time [min].
        """

        v_s = self.calculate_solid_velocity(
            current_rpm,
            fill_degree
        )

        time_sec = self.L / v_s

        return time_sec / 60.0