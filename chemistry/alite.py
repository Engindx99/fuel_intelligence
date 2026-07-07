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

        # ================= STOICHIOMETRY =================

        self.CaO_required=56.08/172.24

        self.C3S_produced=228.32/172.24


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
            state.solids.C2S,
            state.solids.CaO/self.CaO_required
        )


        reacted = self.reacted_mass(
            available,
            rate,
            state.dt,
        )


        # consume C2S
        state.solids.C2S -= reacted


        # consume CaO
        state.solids.CaO -= (
            reacted
            *
            self.CaO_required
        )


        # produce C3S
        state.solids.C3S += (
            reacted
            *
            self.C3S_produced
        )


        state.Alite_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )


        return state