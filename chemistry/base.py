import numpy as np


class ReactionBase:

    def __init__(self):

        # ================= KINETICS =================
        self.prefactor = 0.0
        self.activation_energy = 0.0
        self.R = 8.314

        # ================= THERMODYNAMICS =================
        self.deltaH = 0.0              # J/kg reacted
        self.product_ratio = 0.0       # kg product / kg reacted

        # ================= ACTIVE TEMPERATURE RANGE =================
        self.T_start = 0.0
        self.T_end = 5000.0


    # ======================================================
    # ACTIVE MASK
    # ======================================================
    def active_mask(self, T):

        return (T >= self.T_start) & (T <= self.T_end)


    # ======================================================
    # ARRHENIUS RATE
    # ======================================================
    def reaction_rate(self, T):

        k = self.prefactor * np.exp(
            -self.activation_energy / (self.R * T)
        )

        return np.clip(
            np.where(self.active_mask(T), k, 0.0),
            0.0,
            1.0,
        )


    # ======================================================
    # REACTED MASS FLOW
    # ======================================================
    def reacted_mass_flow(self, m_dot_reactive, reaction_rate):

        return m_dot_reactive * np.mean(reaction_rate)


    # ======================================================
    # HEAT SINK
    # ======================================================
    def heat_sink(self, m_dot_reactive, reaction_rate):

        return (
            self.reacted_mass_flow(
                m_dot_reactive,
                reaction_rate,
            )
            * self.deltaH
        )


    # ======================================================
    # PRODUCT GENERATION
    # ======================================================
    def product_generation(self, m_dot_reactive, reaction_rate):

        return (
            self.reacted_mass_flow(
                m_dot_reactive,
                reaction_rate,
            )
            * self.product_ratio
        )


    # ======================================================
    # CONVERSION UPDATE
    # ======================================================
    def update_conversion(self, X, reaction_rate):

        return np.clip(
            X - reaction_rate,
            0.0,
            1.0,
        )
        
    # ======================================================
    # LIMITING REACTANT
    # ======================================================
    def limiting_mass_flow(self, *mass_flows):

        return np.min(np.asarray(mass_flows), axis=0)