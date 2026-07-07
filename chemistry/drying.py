import numpy as np

from chemistry.base import ReactionBase


class DryingModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 2.0e4
        self.activation_energy = 5.0e4

        # ================= THERMODYNAMICS =================
        self.deltaH = 2.26e6
        self.product_ratio = 1.0

        # ================= TEMPERATURE WINDOW =================
        self.T_start = 300.0
        self.T_end = 473.0


    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        rate = self.reaction_rate(
            state.Ts_calciner
        )

        m_dot_water = (
            state.m_dot_s
            * state.H2O_mass_fraction
        )

        state.Drying_Q_sink = self.heat_sink(
            m_dot_water,
            rate,
        )

        state.m_dot_H2O_drying = self.product_generation(
            m_dot_water,
            rate,
        )

        state.X_H2O = self.update_conversion(
            state.X_H2O,
            rate,
        )

        return state