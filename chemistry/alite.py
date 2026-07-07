import numpy as np

from chemistry.base import ReactionBase


class AliteModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 2.0e3
        self.activation_energy = 2.2e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 6.0e5

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
            state.solids.C2S,
            state.solids.CaO,
        )

        reacted = self.reacted_mass(
            available,
            rate,
            state.dt,
        )

        state.solids.C2S -= reacted

        state.solids.CaO -= reacted

        state.solids.C3S += reacted

        state.Alite_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )

        return state