from dataclasses import dataclass

import numpy as np

from physics.physics import heat_transfer
from physics.physics import h_gas
from physics.physics import wall_losses


@dataclass
class PreheaterStage:
    """
    Common physical model for a single preheater stage.

    The stage contains the existing lumped thermal model:
        gas ↔ solid
        gas ↔ wall
        solid ↔ wall
        wall heat loss
        reaction heat/sink
        energy balance

    NOTE:
        This class intentionally does not model cyclone separation
        or raw-meal transport explicitly yet.
    """

    stage_id: int

    gas_inlet_temperature: float = 0.0
    gas_outlet_temperature: float = 0.0

    gas_inlet_enthalpy: float = 0.0
    gas_outlet_enthalpy: float = 0.0

    solid_inlet_temperature: float = 0.0
    solid_outlet_temperature: float = 0.0

    solid_inlet_enthalpy: float = 0.0
    solid_outlet_enthalpy: float = 0.0

    wall_temperature: float = 0.0

    Q_gs: float = 0.0
    Q_gw: float = 0.0
    Q_ws: float = 0.0

    Q_wall_loss: float = 0.0
    Q_reaction: float = 0.0

    energy_in: float = 0.0
    energy_out: float = 0.0
    energy_residual: float = 0.0

    gas_T_min: float = 0.0
    gas_T_max: float = 0.0

    def __post_init__(self):
        if self.stage_id < 1:
            raise ValueError("stage_id must be >= 1")

    def reset_diagnostics(self):
        self.Q_gs = 0.0
        self.Q_gw = 0.0
        self.Q_ws = 0.0
        self.Q_wall_loss = 0.0
        self.Q_reaction = 0.0

        self.energy_in = 0.0
        self.energy_out = 0.0
        self.energy_residual = 0.0

        self.gas_T_min = 0.0
        self.gas_T_max = 0.0

    def solve(
        self,
        gas_inlet_temperature,
        solid_inlet_temperature,
        m_dot_g,
        m_dot_s,
        state,
        model,
        reaction_power=0.0,
    ):
        """
        Solve one preheater stage.

        IMPORTANT:
            The equations are intentionally kept identical to the
            previous PreheaterStage implementation.
        """

        self.reset_diagnostics()

        eps = model.eps
        T_ref = model.T_ref
        T_amb = model.T_amb

        Tg_in = float(gas_inlet_temperature)
        Ts_in = float(solid_inlet_temperature)

        self.gas_inlet_temperature = Tg_in
        self.solid_inlet_temperature = Ts_in

        # ==========================================================
        # WALL TEMPERATURE SOLUTION
        # ==========================================================

        def wall_residual_stage(Tw_trial):

            _, q_gw_trial, q_ws_trial = heat_transfer(
                Tg=np.array([Tg_in]),
                Ts=np.array([Ts_in]),
                Tw=np.array([Tw_trial]),
                hv_gs=model.hv_gs,
                hv_gw=model.hv_gw,
                hv_ws=model.hv_ws,
                a_gs=model.a_gs,
                a_gw=model.a_gw,
                a_ws=model.a_ws,
                zone=model.zone,
            )

            _, Q_wall_loss_trial, _ = wall_losses(
                Tw=np.array([Tw_trial]),
                h_ext=model.h_ext,
                A_wall_cell=model.A_wall_cell,
                V_cell=model.V_cell,
                T_amb=T_amb,
                A_wall_total=model.A_wall,
                N=1,
                refractory_thickness=model.refractory_thickness,
                refractory_conductivity=model.refractory_conductivity,
                eps=eps,
            )

            return (
                float(q_gw_trial[0] + q_ws_trial[0])
                * model.V_cell
                - Q_wall_loss_trial
            )

        T_low = T_amb
        T_high = max(Tg_in, Ts_in, T_amb)

        residual_high = wall_residual_stage(T_high)

        for _ in range(50):

            if residual_high >= 0.0:
                break

            T_high *= 1.10
            residual_high = wall_residual_stage(T_high)

        else:
            T_high = max(T_amb, Tg_in, Ts_in)

        residual_low = wall_residual_stage(T_low)

        if residual_low * residual_high <= 0.0:

            for _ in range(60):

                T_mid = 0.5 * (T_low + T_high)
                residual_mid = wall_residual_stage(T_mid)

                if abs(residual_mid) < 1e-6:
                    break

                if residual_low * residual_mid <= 0.0:
                    T_high = T_mid
                    residual_high = residual_mid
                else:
                    T_low = T_mid
                    residual_low = residual_mid

            Tw = 0.5 * (T_low + T_high)

        else:

            Tw = max(
                T_amb,
                min(Tg_in, Ts_in),
            )

        self.wall_temperature = float(Tw)

        # ==========================================================
        # HEAT TRANSFER
        # ==========================================================

        _, q_gw, q_ws = heat_transfer(
            Tg=np.array([Tg_in]),
            Ts=np.array([Ts_in]),
            Tw=np.array([Tw]),
            hv_gs=model.hv_gs,
            hv_gw=model.hv_gw,
            hv_ws=model.hv_ws,
            a_gs=model.a_gs,
            a_gw=model.a_gw,
            a_ws=model.a_ws,
            zone=model.zone,
        )

        q_gs, _, _ = heat_transfer(
            Tg=np.array([Tg_in]),
            Ts=np.array([Ts_in]),
            Tw=np.array([Tw]),
            hv_gs=model.hv_gs,
            hv_gw=model.hv_gw,
            hv_ws=model.hv_ws,
            a_gs=model.a_gs,
            a_gw=model.a_gw,
            a_ws=model.a_ws,
            zone=model.zone,
        )

        self.Q_gs = float(q_gs[0]) * model.V_cell
        self.Q_gw = float(q_gw[0]) * model.V_cell
        self.Q_ws = float(q_ws[0]) * model.V_cell

        # ==========================================================
        # WALL LOSS
        # ==========================================================

        _, wall_loss, _ = wall_losses(
            Tw=np.array([Tw]),
            h_ext=model.h_ext,
            A_wall_cell=model.A_wall_cell,
            V_cell=model.V_cell,
            T_amb=T_amb,
            A_wall_total=model.A_wall,
            N=1,
            refractory_thickness=model.refractory_thickness,
            refractory_conductivity=model.refractory_conductivity,
            eps=eps,
        )

        self.Q_wall_loss = float(wall_loss)

        # ==========================================================
        # GAS ENTHALPY
        # ==========================================================

        H_gas_in = m_dot_g * h_gas(
            Tg_in,
            T_ref,
        )

        H_gas_out = (
            H_gas_in
            + reaction_power
            - self.Q_gs
            - self.Q_gw
        )

        Tg_out = model.gas_temperature_from_enthalpy(
            H_gas_out,
            state,
        )

        # ==========================================================
        # SOLID ENTHALPY
        # ==========================================================

        H_solid_in = (
            m_dot_s
            * model.Cp_s
            * (Ts_in - T_ref)
        )

        H_solid_out = (
            H_solid_in
            + self.Q_gs
            - self.Q_ws
        )

        Ts_out = (
            T_ref
            + H_solid_out
            / (m_dot_s * model.Cp_s + eps)
        )

        # ==========================================================
        # ENERGY BALANCE
        # ==========================================================

        energy_in = H_gas_in + H_solid_in

        energy_out = (
            H_gas_out
            + H_solid_out
            + self.Q_wall_loss
            - reaction_power
        )

        energy_residual = energy_in - energy_out

        # ==========================================================
        # STORE RESULTS
        # ==========================================================

        self.gas_outlet_temperature = float(Tg_out)
        self.solid_outlet_temperature = float(Ts_out)

        self.gas_inlet_enthalpy = float(H_gas_in)
        self.gas_outlet_enthalpy = float(H_gas_out)

        self.solid_inlet_enthalpy = float(H_solid_in)
        self.solid_outlet_enthalpy = float(H_solid_out)

        self.Q_reaction = float(reaction_power)

        self.energy_in = float(energy_in)
        self.energy_out = float(energy_out)
        self.energy_residual = float(energy_residual)

        self.gas_T_min = min(Tg_in, Tg_out)
        self.gas_T_max = max(Tg_in, Tg_out)

        return self