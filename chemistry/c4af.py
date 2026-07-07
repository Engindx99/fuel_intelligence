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


        # ================= STOICHIOMETRY =================

        # per kg Al2O3 reacted

        self.CaO_required = (
            (4.0 * 56.08)
            /
            101.96
        )

        self.Fe2O3_required = (
            159.69
            /
            101.96
        )

        self.C4AF_produced = (
            485.97
            /
            101.96
        )


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


        # limiting reactant based on Al2O3
        available = np.minimum.reduce([
            state.solids.Al2O3,
            state.solids.CaO / self.CaO_required,
            state.solids.Fe2O3 / self.Fe2O3_required,
        ])


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


        # consume Fe2O3
        state.solids.Fe2O3 -= (
            reacted
            *
            self.Fe2O3_required
        )


        # produce C4AF
        state.solids.C4AF += (
            reacted
            *
            self.C4AF_produced
        )


        state.C4AF_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )


        return state