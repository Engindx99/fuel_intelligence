import numpy as np


def safe_f(value):

    if isinstance(value, list):
        return float(value[0])

    return float(value)


class EnergyModel:

    def __init__(self, config):

        # ==========================================================
        # BASE CONVECTION
        # ==========================================================

        self.base_h_gs = safe_f(config["gas"]["h_gs"])

        # ==========================================================
        # STEFAN-BOLTZMANN
        # ==========================================================

        self.sigma = 5.670374419e-8

        # ==========================================================
        # EMISSIVITIES
        # ==========================================================

        self.eps_g = safe_f(
            config["gas"].get("emissivity_g", 0.35)
        )

        self.eps_s = safe_f(
            config["material"].get("emissivity_s", 0.85)
        )

        self.eps_w = safe_f(
            config["kiln"].get("emissivity_w", 0.90)
        )

        # Effective gas-solid emissivity
        self.eps_gs = np.sqrt(
            self.eps_g * self.eps_s
        )

        # Effective wall-solid emissivity
        self.eps_ws = 1.0 / (
            1.0 / self.eps_s
            + 1.0 / self.eps_w
            - 1.0
        )

    # ==============================================================
    # CONVECTION
    # ==============================================================

    def calculate_convection_coeff(
        self,
        current_fan_rate,
        nominal_fan=800.0
    ):
        """
        Reynolds-type turbulent scaling.
        """

        fan_ratio = max(
            0.05,
            current_fan_rate / nominal_fan
        )

        return (
            self.base_h_gs
            * (fan_ratio ** 0.67)
        )

    # ==============================================================
    # CONVECTION HEAT FLUX
    # ==============================================================

    def calculate_convective_flux(
        self,
        Tg,
        Ts,
        area,
        h_gs
    ):
        """
        Gas-solid convective transfer.
        """

        q_conv = (
            h_gs
            * area
            * (Tg - Ts)
        )

        return np.nan_to_num(
            q_conv,
            nan=0.0
        )

    # ==============================================================
    # RADIATION
    # ==============================================================

    def calculate_radiation_flux(
        self,
        Tg,
        Ts,
        Tw,
        area
    ):
        """
        Rotary kiln radiation model.

        Components:
        1) Gas -> Solid
        2) Wall -> Solid
        """

        Tg = np.clip(Tg, 300.0, 2600.0)
        Ts = np.clip(Ts, 300.0, 2600.0)
        Tw = np.clip(Tw, 300.0, 2600.0)

        # ----------------------------------------------------------
        # VIEW FACTORS
        # ----------------------------------------------------------

        # Gas fills volume but not perfect enclosure
        f_gs = 0.65

        # Rotating refractory reradiation
        f_ws = 0.75

        # ----------------------------------------------------------
        # GAS -> SOLID
        # ----------------------------------------------------------

        q_gs = (
            f_gs
            * self.eps_gs
            * self.sigma
            * area
            * (Tg**4 - Ts**4)
        )

        # ----------------------------------------------------------
        # WALL -> SOLID
        # ----------------------------------------------------------

        q_ws = (
            f_ws
            * self.eps_ws
            * self.sigma
            * area
            * (Tw**4 - Ts**4)
        )

        # ----------------------------------------------------------
        # TOTAL
        # ----------------------------------------------------------

        q_rad = q_gs + q_ws

        # Radiation limiter
        q_rad = np.clip(
            q_rad,
            -1e7,
            1e7
        )

        return np.nan_to_num(
            q_rad,
            nan=0.0,
            posinf=1e7,
            neginf=-1e7
        )

    # ==============================================================
    # REACTION HEAT
    # ==============================================================

    def get_reaction_heat(
        self,
        rates,
        m_dot_s,
        dH_vec
    ):
        """
        Reaction thermal source term.

        Positive:
            Exothermic heating

        Negative:
            Endothermic cooling
        """

        q_rxn = (
            np.sum(
                rates * dH_vec[:, None],
                axis=0
            )
            * m_dot_s
        )

        return np.nan_to_num(
            q_rxn,
            nan=0.0,
            posinf=1e8,
            neginf=-1e8
        )