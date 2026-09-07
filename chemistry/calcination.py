import numpy as np

from chemistry.base import ReactionBase


class CalcinationModel(ReactionBase):

    def __init__(self):

        super().__init__()

        # ==================================================
        # KINETICS
        # ==================================================
        self.prefactor = 1.0e3          # 1/s
        self.activation_energy = 1.5e5  # J/mol

        # ==================================================
        # THERMODYNAMICS
        # ==================================================
        self.deltaH = 1.78e6            # J/kg CaCO3

        # ==================================================
        # STOICHIOMETRY
        # ==================================================
        self.CaO_ratio = 56.08 / 100.09
        self.CO2_ratio = 44.01 / 100.09

        # ==================================================
        # TEMPERATURE WINDOW
        # ==================================================
        self.T_start = 1073.0           # K
        self.T_end = 1300.0             # K

    # ======================================================
    # STEADY-STATE CALCINATION
    # ======================================================
    def apply(self, state, tau):

        mat = state.materials["calciner"]

        # ======================================================
        # TIME
        # ======================================================

        tau = max(float(tau), 1.0e-9)

        dt = max(
            float(state.dt),
            1.0e-9,
        )

        # ======================================================
        # TEMPERATURE
        # ======================================================

        T = np.maximum(
            np.asarray(
                state.Ts_calciner,
                dtype=float,
            ),
            1.0,
        )

        # ======================================================
        # REACTION RATE
        #
        # k(T) -> 1/s
        # ======================================================

        rate = self.reaction_rate(T)

        # ======================================================
        # CONVERSION
        #
        # X = 1 - exp(-k * tau)
        # ======================================================

        conversion = (
            1.0
            - np.exp(
                -rate * tau
            )
        )

        conversion = np.clip(
            conversion,
            0.0,
            1.0,
        )

        # ======================================================
        # SOLID INVENTORY
        # ======================================================

        CaCO3 = np.maximum(
            np.asarray(
                mat.solids.CaCO3,
                dtype=float,
            ),
            0.0,
        )

        # ======================================================
        # TOTAL SOLID INVENTORY
        # ======================================================

        solid_mass = 0.0

        for name in mat.solids.__dataclass_fields__:

            values = np.asarray(
                getattr(
                    mat.solids,
                    name,
                ),
                dtype=float,
            )

            solid_mass += np.sum(
                np.maximum(
                    values,
                    0.0,
                )
            )

        CaCO3_mass = np.sum(CaCO3)

        CaCO3_fraction = (
            CaCO3_mass
            / max(
                solid_mass,
                1.0e-12,
            )
        )

        CaCO3_fraction = np.clip(
            CaCO3_fraction,
            0.0,
            1.0,
        )

        # ======================================================
        # CaCO3 INLET MASS FLOW
        # ======================================================

        m_dot_CaCO3_total = (
            state.m_dot_s
            * CaCO3_fraction
        )

        # ======================================================
        # DISTRIBUTE CaCO3 FLOW OVER CELLS
        # ACCORDING TO CURRENT INVENTORY
        # ======================================================

        CaCO3_inventory_total = np.sum(CaCO3)

        if CaCO3_inventory_total > 1.0e-12:

            inventory_fraction = (
                CaCO3
                / CaCO3_inventory_total
            )

        else:

            inventory_fraction = (
                np.zeros_like(CaCO3)
            )

        m_dot_CaCO3 = (
            m_dot_CaCO3_total
            * inventory_fraction
        )

        # ======================================================
        # REACTED CaCO3 MASS FLOW
        # ======================================================

        m_dot_reacted = (
            m_dot_CaCO3
            * conversion
        )

        m_dot_reacted_total = np.sum(
            m_dot_reacted
        )

        # ======================================================
        # SAFETY CHECK
        # ======================================================

        if (
            m_dot_reacted_total
            > m_dot_CaCO3_total + 1.0e-9
        ):

            raise ValueError(
                "Calciner: reacted CaCO3 flow "
                "exceeds CaCO3 inlet flow"
            )

        # ======================================================
        # MASS REACTION DURING THIS TIMESTEP
        #
        # kg/s * s = kg
        # ======================================================

        dm_CaCO3 = (
            m_dot_reacted
            * dt
        )

        # ======================================================
        # LIMIT REACTION TO AVAILABLE INVENTORY
        # ======================================================

        dm_CaCO3 = np.minimum(
            dm_CaCO3,
            CaCO3,
        )

        dm_CaCO3_total = np.sum(
            dm_CaCO3
        )

        # ======================================================
        # STOICHIOMETRIC PRODUCTS
        #
        # CaCO3 -> CaO + CO2
        # ======================================================

        dm_CaO = (
            dm_CaCO3
            * self.CaO_ratio
        )

        dm_CO2 = (
            dm_CaCO3
            * self.CO2_ratio
        )

        # ======================================================
        # UPDATE SOLID PHASES
        # ======================================================

        mat.solids.CaCO3 -= dm_CaCO3

        mat.solids.CaO += dm_CaO

        # ======================================================
        # UPDATE GAS PHASE
        # ======================================================

        mat.gases.CO2 += dm_CO2

        # ======================================================
        # NUMERICAL CLEANUP
        # ======================================================

        mat.solids.CaCO3 = np.maximum(
            mat.solids.CaCO3,
            0.0,
        )

        # ======================================================
        # REACTION HEAT
        #
        # kg/s * J/kg = W
        # ======================================================

        Q_calcination = (
            m_dot_reacted
            * self.deltaH
        )

        Q_calcination = np.sum(
            Q_calcination
        )

        # ======================================================
        # STATE OUTPUTS
        # ======================================================

        state.Calcination_Q_sink = float(
            Q_calcination
        )

        state.Calciner_Q_sink = float(
            Q_calcination
        )

        state.X_CaCO3_calciner = float(
            np.mean(conversion)
        )

        state.X_calcination = float(
            np.mean(conversion)
        )

        state.tau_calciner = float(
            tau
        )

        state.m_dot_CaCO3_reacted_calciner = float(
            m_dot_reacted_total
        )

        # ======================================================
        # MASS REACTION DEBUG
        # ======================================================

        state.m_dot_CaO_generated_calciner = float(
            np.sum(dm_CaO) / dt
        )

        state.m_dot_CO2_generated_calciner = float(
            np.sum(dm_CO2) / dt
        )

        state.dm_CaCO3_reacted_calciner = float(
            dm_CaCO3_total
        )

        state.dm_CaO_generated_calciner = float(
            np.sum(dm_CaO)
        )

        state.dm_CO2_generated_calciner = float(
            np.sum(dm_CO2)
        )

        return state