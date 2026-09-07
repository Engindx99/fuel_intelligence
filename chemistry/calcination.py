import numpy as np

from chemistry.base import ReactionBase
from chemistry.composition import RAW_MEAL_COMPOSITION


class CalcinationModel(ReactionBase):

    def __init__(self):
        super().__init__()

        # ======================================================
        # KINETICS
        # ======================================================

        self.prefactor = 12.0             # 1/s
        self.activation_energy = 1.5e5      # J/mol
        self.deltaH = 1.78e6                # J/kg CaCO3

        # ======================================================
        # STOICHIOMETRY
        # CaCO3 -> CaO + CO2
        # ======================================================

        self.CaO_ratio = 56.08 / 100.09
        self.CO2_ratio = 44.01 / 100.09

        # ======================================================
        # TEMPERATURE WINDOW
        # ======================================================

        self.T_start = 1073.0               # K
        self.T_end = 1300.0                 # K

        # ======================================================
        # RAW MEAL COMPOSITION
        # ======================================================

        self.CaCO3_mass_fraction = (
            RAW_MEAL_COMPOSITION["CaCO3"] / 100000.0
        )

    # ======================================================
    # STEADY-STATE SPATIAL REACTION
    # ======================================================

    def apply(self, state, dz, u_s):

        # ==================================================
        # INPUTS
        # ==================================================

        T = np.asarray(
            state.Ts_calciner,
            dtype=float,
        )

        if T.ndim != 1:
            raise ValueError(
                "Ts_calciner must be a 1D array."
            )

        N = len(T)

        if N == 0:
            raise ValueError(
                "Calciner reaction requires at least one cell."
            )

        dz = float(dz)

        if not np.isfinite(dz) or dz <= 0.0:
            raise ValueError(
                "Calciner spatial step dz must be > 0."
            )

        u_s = float(u_s)

        if not np.isfinite(u_s) or u_s <= 0.0:
            raise ValueError(
                "Calciner solid velocity u_s must be > 0."
            )

        m_dot_s = max(
            float(state.m_dot_s),
            0.0,
        )

        # ==================================================
        # INLET CaCO3 MASS FLOW
        # ==================================================

        m_dot_CaCO3_in = (
            m_dot_s
            * self.CaCO3_mass_fraction
        )

        # ==================================================
        # SPATIAL ARRAYS
        # ==================================================

        m_dot_CaCO3_in_cells = np.zeros(N)
        m_dot_CaCO3_reacted_cells = np.zeros(N)
        m_dot_CaCO3_out_cells = np.zeros(N)

        reaction_rate_cells = np.zeros(N)
        reaction_heat_cells = np.zeros(N)
        conversion_cells = np.zeros(N)

        # ==================================================
        # INITIAL CONDITION
        #
        # Solid flows from cell 0 -> cell N-1
        # ==================================================

        m_dot_CaCO3_in_cells[0] = (
            m_dot_CaCO3_in
        )

        # ==================================================
        # SPACE MARCHING
        #
        # u_s * dm/dz = -k(T) * m
        #
        # dm/dz = -(k/u_s) * m
        #
        # Exact cell integration:
        #
        # m_out = m_in * exp[-k*dz/u_s]
        # ==================================================

        for i in range(N):

            m_in = max(
                float(m_dot_CaCO3_in_cells[i]),
                0.0,
            )

            T_i = max(
                float(T[i]),
                1.0,
            )

            # --------------------------------------------------
            # LOCAL KINETIC RATE
            # --------------------------------------------------

            k_i = float(
                self.reaction_rate(T_i)
            )

            reaction_rate_cells[i] = k_i

            # --------------------------------------------------
            # SPATIAL REACTION EXPONENT
            # --------------------------------------------------

            spatial_exponent = (
                k_i
                * dz
                / u_s
            )

            # --------------------------------------------------
            # OUTLET CaCO3 FLOW
            # --------------------------------------------------

            m_out = (
                m_in
                * np.exp(-spatial_exponent)
            )

            m_out = np.clip(
                m_out,
                0.0,
                m_in,
            )

            # --------------------------------------------------
            # REACTED CaCO3 FLOW
            # --------------------------------------------------

            m_reacted = (
                m_in
                - m_out
            )

            # --------------------------------------------------
            # LOCAL CONVERSION
            # --------------------------------------------------

            if m_in > 1.0e-12:
                X_i = (
                    m_reacted
                    / m_in
                )
            else:
                X_i = 0.0

            X_i = np.clip(
                X_i,
                0.0,
                1.0,
            )

            # --------------------------------------------------
            # STORE CELL RESULTS
            # --------------------------------------------------

            m_dot_CaCO3_reacted_cells[i] = (
                m_reacted
            )

            m_dot_CaCO3_out_cells[i] = (
                m_out
            )

            conversion_cells[i] = X_i

            reaction_heat_cells[i] = (
                m_reacted
                * self.deltaH
            )

            # --------------------------------------------------
            # NEXT CELL
            # --------------------------------------------------

            if i + 1 < N:

                m_dot_CaCO3_in_cells[i + 1] = (
                    m_out
                )

        # ==================================================
        # TOTAL REACTION
        # ==================================================

        m_dot_CaCO3_reacted = np.sum(
            m_dot_CaCO3_reacted_cells
        )

        m_dot_CaCO3_out = (
            m_dot_CaCO3_in
            - m_dot_CaCO3_reacted
        )

        m_dot_CaCO3_out = max(
            float(m_dot_CaCO3_out),
            0.0,
        )

        # ==================================================
        # PRODUCTS
        # ==================================================

        m_dot_CaO_generated = (
            m_dot_CaCO3_reacted
            * self.CaO_ratio
        )

        m_dot_CO2_generated = (
            m_dot_CaCO3_reacted
            * self.CO2_ratio
        )

        # ==================================================
        # TOTAL CALCINATION HEAT
        # ==================================================

        Q_calcination = np.sum(
            reaction_heat_cells
        )

        # ==================================================
        # OVERALL CONVERSION
        # ==================================================

        X_total = (
            m_dot_CaCO3_reacted
            / max(
                m_dot_CaCO3_in,
                1.0e-12,
            )
        )

        X_total = np.clip(
            X_total,
            0.0,
            1.0,
        )

        # ==================================================
        # STATE OUTPUTS
        # ==================================================

        state.Calcination_Q_sink = float(
            Q_calcination
        )

        state.Calciner_Q_sink = float(
            Q_calcination
        )

        state.X_CaCO3_calciner = float(
            X_total
        )

        state.X_calcination = float(
            X_total
        )

        # ==================================================
        # MASS FLOWS
        # ==================================================

        state.m_dot_CaCO3_in_calciner = float(
            m_dot_CaCO3_in
        )

        state.m_dot_CaCO3_reacted_calciner = float(
            m_dot_CaCO3_reacted
        )

        state.m_dot_CaCO3_out_calciner = float(
            m_dot_CaCO3_out
        )

        state.m_dot_CaO_generated_calciner = float(
            m_dot_CaO_generated
        )

        state.m_dot_CO2_generated_calciner = float(
            m_dot_CO2_generated
        )

        # ==================================================
        # CELL RESULTS
        # ==================================================

        state.X_CaCO3_cells = (
            conversion_cells.copy()
        )

        state.reaction_rate_CaCO3_cells = (
            reaction_rate_cells.copy()
        )

        state.m_dot_CaCO3_in_cells = (
            m_dot_CaCO3_in_cells.copy()
        )

        state.m_dot_CaCO3_reacted_cells = (
            m_dot_CaCO3_reacted_cells.copy()
        )

        state.m_dot_CaCO3_out_cells = (
            m_dot_CaCO3_out_cells.copy()
        )

        state.Calcination_Q_cells = (
            reaction_heat_cells.copy()
        )

        # ==================================================
        # STEADY-STATE MASS BALANCE
        # ==================================================

        state.CaCO3_steady_state_balance = float(
            m_dot_CaCO3_in
            - m_dot_CaCO3_reacted
            - m_dot_CaCO3_out
        )

        # ==================================================
        # NUMERICAL CHECKS
        # ==================================================

        if not np.all(
            np.isfinite(
                m_dot_CaCO3_out_cells
            )
        ):
            raise FloatingPointError(
                "Non-finite CaCO3 spatial solution."
            )

        if not np.all(
            np.isfinite(
                reaction_heat_cells
            )
        ):
            raise FloatingPointError(
                "Non-finite calcination heat distribution."
            )

        if not np.isfinite(Q_calcination):
            raise FloatingPointError(
                "Non-finite total calcination heat."
            )

        return state