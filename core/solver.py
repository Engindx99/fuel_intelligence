import numpy as np
from core.state import KilnState


class KilnSolver:

    def __init__(self, config, kinetics_fn, transport, energy):

        self.cfg = config
        self.kin = kinetics_fn
        self.tra = transport
        self.en = energy

        nodes = int(self._safe_f(config["kiln"]["nodes"]))
        self.state = KilnState(nodes)

        self.dz = self._safe_f(config["kiln"]["length"]) / nodes
        self.total_sim_time = 0.0

        self.state.initialize_profiles(
            T_ambient=300.0,
            T_gas_inlet=2000.0,
            raw_meal_comp=config["raw_meal_composition"]
        )

    # ==========================================================
    # UTIL
    # ==========================================================

    def _safe_f(self, v):
        return float(v[0]) if isinstance(v, list) else float(v)

    # ==========================================================
    # KINETICS PARAMS
    # ==========================================================

    def _get_kinetics_params(self):

        k = self.cfg["kinetics"]

        k0_vec = np.array([
            k["k0"],
            k["k0_c2s"],
            k["k0_c3s"],
            k["pre_factor_c3a"],
            k["pre_factor_c4af"]
        ], dtype=np.float64)

        Ea_vec = np.array([
            k["Ea"],
            k["Ea_c2s"],
            k["Ea_c3s"],
            0.0,
            0.0
        ], dtype=np.float64)

        T_min_vec = np.array([
            float(k["T_min_rxn"]),
            float(k["T_min_c2s"]),
            float(k["T_min_c3s"]),
            1350.0,
            1350.0
        ], dtype=np.float64)

        pre_factors = np.array([
            1.0,
            1.0,
            1.0,
            k["pre_factor_c3a"],
            k["pre_factor_c4af"]
        ], dtype=np.float64)

        return k0_vec, Ea_vec, T_min_vec, pre_factors

    # ==========================================================
    # MAIN STEP
    # ==========================================================

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):

        dt = min(float(dt), 0.25)

        cp_s = self._safe_f(self.cfg["material"]["cp_s"])
        cp_g = self._safe_f(self.cfg["gas"]["cp_g"])
        diameter = self._safe_f(self.cfg["kiln"]["diameter"])

        # ------------------------------------------------------
        # FLOW
        # ------------------------------------------------------

        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        fill = self.tra.get_dynamic_filling_degree(kiln_rpm)

        area_total = np.pi * (diameter * 0.5) ** 2
        area_gas = area_total * (1.0 - fill)
        exchange_area = np.pi * diameter * self.dz

        # ------------------------------------------------------
        # MASS FLOW
        # ------------------------------------------------------

        m_dot_s = (feed_rate * 1000.0) / 3600.0

        m_dot_g = (
            self._safe_f(self.cfg["gas"]["nominal_flow"])
            * (fan_rate / 800.0)
        )

        self.state.rho_g = np.maximum(self.state.rho_g, 0.1)

        # ======================================================
        # ADVECTION (SOLID)
        # ======================================================

        cfl = np.clip(v_s * dt / self.dz, 0.0, 0.2)

        fields = [
            "CaCO3", "CaO", "SiO2",
            "Al2O3", "Fe2O3",
            "C2S", "C3S", "C3A", "C4AF"
        ]

        for f in fields:
            arr = getattr(self.state, f)
            arr[1:] = (1 - cfl) * arr[1:] + cfl * arr[:-1]

        self.state.Ts[1:] = (1 - cfl) * self.state.Ts[1:] + cfl * self.state.Ts[:-1]

        # ======================================================
        # GAS FLOW
        # ======================================================

        gas_cfl = np.clip(
            (m_dot_g / (np.maximum(0.1, self.state.rho_g) * area_gas))
            * dt / self.dz,
            0.0, 0.2
        )

        self.state.Tg[:-1] = (
            (1 - gas_cfl[:-1]) * self.state.Tg[:-1]
            + gas_cfl[:-1] * self.state.Tg[1:]
        )

        self.state.Tg[-1] = (
            400.0
            + (1 - np.exp(-self.total_sim_time / 28800.0)) * 1600.0
        )

        # ======================================================
        # WALL
        # ======================================================

        Tw_target = 0.55 * self.state.Tg + 0.45 * self.state.Ts
        self.state.Tw += (Tw_target - self.state.Tw) * 0.015 * dt

        # ======================================================
        # HEAT TRANSFER
        # ======================================================

        h_gs = self.en.calculate_convection_coeff(fan_rate)

        q_conv = self.en.calculate_convective_flux(
            self.state.Tg,
            self.state.Ts,
            exchange_area,
            h_gs
        )

        q_rad = self.en.calculate_radiation_flux(
            self.state.Tg,
            self.state.Ts,
            self.state.Tw,
            exchange_area
        )

        # ======================================================
        # KINETICS
        # ======================================================

        k0_vec, Ea_vec, T_min_vec, pre_factors = self._get_kinetics_params()

        rates = self.kin(
        self.state.Ts,
        self.state.CaCO3,
        self.state.CaO,
        self.state.SiO2,
        self.state.C2S,
        self.state.Al2O3,
        self.state.Fe2O3,
        self.state.C3A,
        self.state.C4AF,
        k0_vec,
        Ea_vec,
        self.cfg["kinetics"]["R"],
        T_min_vec,
        pre_factors, 
        dt
    )

        # ======================================================
        # ENERGY (STABLE SIMPLE MODEL)
        # ======================================================

        dH = np.sum(rates) * m_dot_s

        eff_mass = np.maximum(
            5.0,
            (m_dot_s / max(v_s, 1e-4)) * self.dz
        )

        Cp = eff_mass * cp_s

        q_net = q_conv + q_rad - dH

        dTs = q_net / np.maximum(100.0, Cp)
        dTg = -(q_conv + q_rad) / np.maximum(1.0, m_dot_g * cp_g)

        self.state.Ts += np.clip(dTs, -250, 250) * dt
        self.state.Tg += np.clip(dTg, -400, 400) * dt

        self.state.Ts = np.clip(self.state.Ts, 300, 1900)
        self.state.Tg = np.clip(self.state.Tg, 300, 2400)

        # ======================================================
        # CHEMISTRY UPDATE
        # ======================================================

        r = rates * dt

        r0 = np.minimum(r[0], self.state.CaCO3)
        self.state.CaCO3 -= r0
        self.state.CaO += r0 * 0.56

        r1 = np.minimum(r[1], self.state.CaO / 2.0)
        r1 = np.minimum(r1, self.state.SiO2)

        self.state.CaO -= 2 * r1
        self.state.SiO2 -= r1
        self.state.C2S += r1

        r2 = np.minimum(r[2], self.state.C2S)
        r2 = np.minimum(r2, self.state.CaO)

        self.state.C2S -= r2
        self.state.CaO -= r2
        self.state.C3S += r2

        r3 = np.minimum(r[3], self.state.Al2O3)
        r4 = np.minimum(r[4], self.state.Fe2O3)

        self.state.Al2O3 -= r3
        self.state.Fe2O3 -= r4

        # ======================================================
        # FEED
        # ======================================================

        feed = dt * m_dot_s / max(self.state.N, 1)
        raw = self.cfg["raw_meal_composition"]

        self.state.Ts[0] = self._safe_f(self.cfg["material"]["temp_inlet"])

        self.state.CaCO3[0] += raw["CaCO3"] * feed
        self.state.SiO2[0] += raw["SiO2"] * feed
        self.state.Al2O3[0] += raw["Al2O3"] * feed
        self.state.Fe2O3[0] += raw["Fe2O3"] * feed

        # CLEAN
        for arr in [
            self.state.CaCO3, self.state.CaO,
            self.state.SiO2, self.state.Al2O3,
            self.state.Fe2O3,
            self.state.C2S, self.state.C3S,
            self.state.C3A, self.state.C4AF
        ]:
            np.maximum(arr, 0.0, out=arr)

        self.total_sim_time += dt

        return dt