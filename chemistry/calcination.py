import numpy as np

from chemistry.base import ReactionBase


class CalcinationModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 1.0e3
        self.activation_energy = 1.5e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 1.78e6

        # ================= STOICHIOMETRY =================
        self.CaO_ratio = 56.08 / 100.09
        self.CO2_ratio = 44.01 / 100.09

        # ================= TEMPERATURE =================
        self.T_start = 1073.0
        self.T_end = 1300.0

    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        # ======================================================
        # MATERIAL INVENTORY OF CALCINER
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
            mat.solids.CaCO3,
            rate,
            state.dt,
        )


        # ======================================================
        # UPDATE SOLID PHASES
        # ======================================================
        mat.solids.CaCO3 -= reacted

        mat.solids.CaO += (
            reacted
            * self.CaO_ratio
        )

        # ======================================================
        # UPDATE GAS PHASES
        # ======================================================
        mat.gases.CO2 += (
            reacted
            * self.CO2_ratio
        )

        # ======================================================
        # REACTION HEAT
        # ======================================================
        state.Calcination_Q_sink = np.sum(
            self.heat_sink(
                reacted
            )
        )

        return state