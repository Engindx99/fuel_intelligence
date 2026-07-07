import numpy as np

from chemistry.base import ReactionBase


class BeliteModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 2.0e3
        self.activation_energy = 2.0e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 5.0e5

        # ================= TEMPERATURE =================
        self.T_start = 1123.0
        self.T_end = 1473.0

    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        rate = self.reaction_rate(
            state.Ts_burning
        )

        available = np.minimum(
            state.solids.CaO / 2.0,
            state.solids.SiO2,
        )

        reacted = self.reacted_mass(
            available,
            rate,
            state.dt,
        )

        state.solids.CaO -= 2.0 * reacted

        state.solids.SiO2 -= reacted

        state.solids.C2S += reacted

        state.Belite_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )

        return state