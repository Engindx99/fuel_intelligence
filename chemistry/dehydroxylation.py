import numpy as np


class DehydroxylationModel:

    def __init__(self):

        # ======================================================
        # KINETICS
        # ======================================================
        self.prefactor = 5.0e3
        self.activation_energy = 1.20e5
        self.R = 8.314

        # ======================================================
        # REACTION
        # ======================================================
        self.deltaH = 1.10e6          # J/kg reacted mineral
        self.H2O_ratio = 0.139        # kg H2O / kg reacted clay

        self.T_start = 723.0          # K
        self.T_end = 973.0            # K

    # ======================================================
    # REACTION RATE
    # ======================================================
    def reaction_rate(self, Ts):

        active = (
            (Ts >= self.T_start)
            &
            (Ts <= self.T_end)
        )

        k = (
            self.prefactor
            *
            np.exp(
                -self.activation_energy
                /
                (self.R * Ts)
            )
        )

        r = np.where(
            active,
            k,
            0.0,
        )

        return np.clip(r, 0.0, 1.0)

    # ======================================================
    # HEAT SINK
    # ======================================================
    def heat_sink(self, state, reaction_rate):

        m_dot_clay = (
            state.m_dot_s
            *
            state.OH_mass_fraction
        )

        r_mean = np.mean(reaction_rate)

        m_dot_reacted = (
            m_dot_clay
            *
            r_mean
        )

        return (
            m_dot_reacted
            *
            self.deltaH
        )

    # ======================================================
    # WATER GENERATION
    # ======================================================
    def water_generation(self, state, reaction_rate):

        m_dot_clay = (
            state.m_dot_s
            *
            state.OH_mass_fraction
        )

        r_mean = np.mean(reaction_rate)

        m_dot_reacted = (
            m_dot_clay
            *
            r_mean
        )

        return (
            m_dot_reacted
            *
            self.H2O_ratio
        )

    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        reaction_rate = self.reaction_rate(
            state.Ts_calciner
        )

        state.Dehydroxylation_Q_sink = self.heat_sink(
            state,
            reaction_rate,
        )

        state.m_dot_H2O_dehydroxylation = (
            self.water_generation(
                state,
                reaction_rate,
            )
        )

        state.X_OH = np.clip(
            state.X_OH
            - reaction_rate,
            0.0,
            1.0,
        )

        return state