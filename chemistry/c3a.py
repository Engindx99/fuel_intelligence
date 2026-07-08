import numpy as np

from chemistry.base import ReactionBase


class C3AModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 1.0e3
        self.activation_energy = 2.2e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 3.0e5


        # ================= STOICHIOMETRY =================

        self.CaO_required=168.24/101.96

        self.C3A_produced=270.16/101.96


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


        # limiting reactant
        available=np.minimum(
            state.solids.Al2O3,
            state.solids.CaO/self.CaO_required
        )


        reacted = self.reacted_mass(
            available,
            rate,
            state.dt,
        )


        # consume Al2O3
        state.solids.Al2O3 -= reacted


        # consume CaO
        state.solids.CaO -= (
            reacted
            *
            self.CaO_required
        )


        # produce C3A
        state.solids.C3A += (
            reacted
            *
            self.C3A_produced
        )


        state.C3A_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )


        return state