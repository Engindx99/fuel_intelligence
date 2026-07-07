import numpy as np

from chemistry.base import ReactionBase


class C3AModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 3.0e3
        self.activation_energy = 2.2e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 3.0e5

        # ================= TEMPERATURE =================
        self.T_start = 1373.0
        self.T_end = 1723.0

    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        rate = self.reaction_rate(
            state.Ts_burning
        )

        available = np.minimum(
            state.solids.CaO / 3.0,
            state.solids.Al2O3,
        )

        reacted = self.reacted_mass(
            available,
            rate,
            state.dt,
        )

        state.solids.CaO -= 3.0 * reacted

        state.solids.Al2O3 -= reacted

        state.solids.C3A += reacted

        state.C3A_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )

        return state