import numpy as np

from chemistry.base import ReactionBase


class C4AFModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ================= KINETICS =================
        self.prefactor = 1.0e3
        self.activation_energy = 2.2e5

        # ================= THERMODYNAMICS =================
        self.deltaH = 3.5e5

        # ================= STOICHIOMETRY =================
        # per kg Al2O3 reacted

        self.CaO_required = (
            (4.0 * 56.08)
            / 101.96
        )

        self.Fe2O3_required = (
            159.69
            / 101.96
        )

        self.C4AF_produced = (
            485.97
            / 101.96
        )

        # ================= TEMPERATURE =================
        self.T_start = 1373.0
        self.T_end = 1723.0


    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state):

        # ======================================================
        # MATERIAL INVENTORY (BURNING)
        # ======================================================
        mat = state.materials["burning"]

        # ======================================================
        # REACTION RATE
        # ======================================================
        rate = self.reaction_rate(
            state.Ts_burning
        )

        # ======================================================
        # LIMITING REACTANT
        # ======================================================
        available = np.minimum.reduce([
            mat.solids.Al2O3,
            mat.solids.CaO / self.CaO_required,
            mat.solids.Fe2O3 / self.Fe2O3_required,
        ])

        # ======================================================
        # REACTED MASS
        # ======================================================
        reacted = self.reacted_mass(
            available,
            rate,
            state.dt,
        )

        # ======================================================
        # UPDATE SOLID PHASES
        # ======================================================
        mat.solids.Al2O3 -= reacted

        mat.solids.CaO -= (
            reacted
            * self.CaO_required
        )

        mat.solids.Fe2O3 -= (
            reacted
            * self.Fe2O3_required
        )

        mat.solids.C4AF += (
            reacted
            * self.C4AF_produced
        )

        # ======================================================
        # REACTION HEAT
        # ======================================================
        state.C4AF_Q_sink = np.sum(
            self.heat_sink(
                reacted,
            )
        )

        return state