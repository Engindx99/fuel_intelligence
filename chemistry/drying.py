import numpy as np

from chemistry.base import ReactionBase


class DryingModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 1.0e3
        self.activation_energy = 5.0e4

        # ================= THERMODYNAMICS =================
        self.deltaH = 2.26e6

        # ================= TEMPERATURE =================
        self.T_start = 300.0
        self.T_end = 473.0

    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        # ======================================================
        # MATERIAL INVENTORY (PREHEATER)
        # ======================================================
        mat = state.materials["preheater"]

        # ======================================================
        # REACTION RATE
        # ======================================================
        rate = self.reaction_rate(
            state.Ts_preheater
        )

        # ======================================================
        # REACTED MASS
        # ======================================================
        reacted = self.reacted_mass(
            mat.solids.H2O,
            rate,
            state.dt,
        )

        # ======================================================
        # UPDATE PHASES
        # ======================================================
        mat.solids.H2O -= reacted

        mat.gases.H2O += reacted

        # ======================================================
        # REACTION HEAT
        # ======================================================
        state.Drying_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )

        return state