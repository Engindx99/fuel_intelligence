import numpy as np

from chemistry.base import ReactionBase


class BeliteModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 2.0e3
        self.activation_energy = 2.0e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 5.0e5          # J/kg C2S formed

        # ================= TEMPERATURE =================
        self.T_start = 1123.0
        self.T_end = 1473.0

    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        # ================= REACTION RATE =================
        rate = self.reaction_rate(
            state.Ts_burning
        )

        # ================= AVAILABLE REACTANTS =================
        m_CaO = (
            state.m_dot_s
            * state.CaO_mass_fraction
        )

        m_SiO2 = (
            state.m_dot_s
            * state.SiO2_mass_fraction
        )

        # ================= LIMITING REACTANT =================
        m_dot_reactive = self.limiting_mass_flow(
            m_CaO,
            m_SiO2,
        )

        # ================= HEAT SINK =================
        state.Belite_Q_sink = self.heat_sink(
            m_dot_reactive,
            rate,
        )

        # ======================================================
        # PLACEHOLDER
        # Full stoichiometric phase update will be implemented
        # after the complete clinker chemistry is available.
        # ======================================================

        return state