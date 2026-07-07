import numpy as np

from chemistry.base import ReactionBase


class C4AFModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 3.0e3
        self.activation_energy = 2.2e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 3.5e5

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

        available = np.minimum.reduce([
            state.solids.CaO / 4.0,
            state.solids.Al2O3,
            state.solids.Fe2O3,
        ])

        reacted = self.reacted_mass(
            available,
            rate,
            state.dt,
        )

        state.solids.CaO -= 4.0 * reacted

        state.solids.Al2O3 -= reacted

        state.solids.Fe2O3 -= reacted

        state.solids.C4AF += reacted

        state.C4AF_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )

        return state