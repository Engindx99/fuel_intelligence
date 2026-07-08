import numpy as np

from chemistry.base import ReactionBase


class DehydroxylationModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 1.0e3
        self.activation_energy = 1.20e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 1.10e6

        # kg H2O released / kg reacted hydroxyl
        self.product_ratio = 0.139

        # ================= TEMPERATURE =================
        self.T_start = 723.0
        self.T_end = 973.0


    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        # ======================================================
        # MATERIAL INVENTORY (CALCINER)
        # ======================================================
        mat = state.materials["calciner"]

        # ======================================================
        # REACTION RATE
        # ======================================================
        rate = self.reaction_rate(
            state.Ts_calciner
        )

        # ======================================================
        # REACTED MASS
        # ======================================================
        reacted = self.reacted_mass(
            mat.solids.Bound_H2O,
            rate,
            state.dt,
        )

        # ======================================================
        # UPDATE SOLID PHASES
        # ======================================================
        mat.solids.Bound_H2O -= reacted

        # ======================================================
        # UPDATE GAS PHASES
        # ======================================================
        mat.gases.H2O += (
            reacted
            * self.product_ratio
        )

        # ======================================================
        # REACTION HEAT
        # ======================================================
        state.Dehydroxylation_Q_sink = np.sum(
            self.heat_sink(
                reacted
            )
        )

        return state