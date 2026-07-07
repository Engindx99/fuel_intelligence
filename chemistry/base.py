import numpy as np


class ReactionBase:

    def __init__(self):

        self.R = 8.314

        # ================= KINETICS =================
        self.prefactor = 1.0
        self.activation_energy = 0.0

        # ================= THERMODYNAMICS =================
        self.deltaH = 0.0

        # ================= TEMPERATURE =================
        self.T_start = 300.0
        self.T_end = 2000.0

    # ======================================================
    # TEMPERATURE WINDOW
    # ======================================================
    def temperature_window(self, T):

        return np.clip(
            (T - self.T_start)
            / (self.T_end - self.T_start + 1e-12),
            0.0,
            1.0,
        )

    # ======================================================
    # ARRHENIUS RATE
    # ======================================================
    def reaction_rate(self, T):

        arrhenius = (
            self.prefactor
            * np.exp(
                -self.activation_energy
                / (self.R * np.maximum(T, 1.0))
            )
        )

        return (
            arrhenius
            * self.temperature_window(T)
        )

    # ======================================================
    # REACTED MASS
    # ======================================================
    def reacted_mass(
        self,
        available,
        rate,
        dt,
    ):

        reacted = (
            available
            * rate
            * dt
        )

        return np.minimum(
            reacted,
            available,
        )

    # ======================================================
    # REACTION HEAT
    # ======================================================
    def heat_sink(
        self,
        reacted,
    ):

        return (
            reacted
            * self.deltaH
        )