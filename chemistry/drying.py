import numpy as np


class DryingModel:

    def __init__(self, N=5, dz=1.0):

        self.N = N
        self.dz = dz

        # ======================================================
        # NUMERICAL
        # ======================================================
        self.eps = 1e-9

        # ======================================================
        # CONSTANTS
        # ======================================================
        self.R = 8.314

        # ======================================================
        # DRYING KINETICS
        # ======================================================
        self.A = 5.0e3
        self.Ea = 2.0e4

        # ======================================================
        # WATER
        # ======================================================
        self.deltaH_evap = 2.26e6

        # ======================================================
        # BUFFERS
        # ======================================================
        self._dX_dz = np.zeros(N)

    # ======================================================
    # DRYING RATE
    # ======================================================
    def rate(self, Ts, X_H2O):

        k = self.A * np.exp(-self.Ea / (self.R * Ts + self.eps))
        reaction_rate = k * X_H2O

        return reaction_rate

    # ======================================================
    # SOLVE
    # ======================================================
    def solve(self, Ts, X_H2O, u_s, dt):

        dX_dz = self._dX_dz

        dX_dz[1:] = (X_H2O[1:] - X_H2O[:-1]) / self.dz
        dX_dz[0] = dX_dz[1]

        reaction_rate = self.rate(Ts, X_H2O)

        X_new = X_H2O + dt * (-u_s * dX_dz - reaction_rate)

        np.clip(X_new, 0.0, 1.0, out=X_new)

        return X_new, reaction_rate

    # ======================================================
    # DRYING HEAT SINK
    # ======================================================
    def heat_sink(self, state, reaction_rate):

        m_dot_water = state.m_dot_s * state.Moisture_mass_fraction

        r_mean = np.mean(reaction_rate)

        m_dot_evaporated = m_dot_water * r_mean

        Q_sink = m_dot_evaporated * self.deltaH_evap

        return Q_sink

    # ======================================================
    # WATER GENERATION
    # ======================================================
    def gas_generation(self, state, reaction_rate):

        m_dot_water = state.m_dot_s * state.Moisture_mass_fraction

        r_mean = np.mean(reaction_rate)

        m_dot_H2O = m_dot_water * r_mean

        return m_dot_H2O