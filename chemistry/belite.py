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

        # ================= STOICHIOMETRY =================
        
        self.CaO_required = 112.16/60.08

        self.C2S_produced = 172.24/60.08

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


        # limiting reactant (SiO2 basis)
        available=np.minimum(
            state.solids.SiO2,
            state.solids.CaO/self.CaO_required
        )


        reacted = self.reacted_mass(
            available,
            rate,
            state.dt,
        )


        # consume reactants
        state.solids.SiO2 -= reacted

        state.solids.CaO -= (
            reacted
            *
            self.CaO_required
        )


        # produce C2S
        state.solids.C2S += (
            reacted
            *
            self.C2S_produced
        )


        state.Belite_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )


        return state