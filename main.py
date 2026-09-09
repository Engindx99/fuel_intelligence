from pyroprocess.globalstate import GlobalState
from pyroprocess.burning import Burning
from pyroprocess.transition import Transition
from pyroprocess.calciner import Calciner
from pyroprocess.preheater import Preheater
from pyroprocess.cooler import Cooler

from controls.mpc import MasterMPC
from physics.mass_transport import MassTransport
from physics.physics import gas_mass_balance
from dataclasses import fields
from chemistry.phases import SolidPhases, GasPhases

from validators.energy_validator import validate_energy
from validators.reporter import report_validation

import numpy as np
import yaml 





def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Twin:

    def __init__(self, state, cfg, mpc_cfg):

        self.state = state
        
        self.mass_transport = MassTransport()

        # ======================================================
        # ZONE MODELS
        # ======================================================
        self.burning = Burning(
            N=cfg["plant"]["N"],
            L=cfg["plant"]["length"],
        )

        self.transition = Transition(
            N=cfg["plant"]["N"],
            L=cfg["transition"]["length"],
        )

        self.calciner = Calciner(
            N=cfg["plant"]["N"],
            L=cfg["calciner"]["length"],
        )

        self.preheater = Preheater(
            N=cfg["plant"]["N"],
            L=cfg["preheater"]["length"],
        )

        self.cooler = Cooler(
            N=cfg["plant"]["N"],
            L=cfg["cooler"]["length"],
        )
        
        # ======================================================
        # NUMERICAL
        # ======================================================
        self.eps = 1e-9

        # ======================================================
        # STEADY-STATE TRANSPORT VARIABLES
        # ======================================================
        self.state.u_s = self.calciner.u_s


        # ======================================================
        # MPC (optional)
        # ======================================================
        # self.mpc = MasterMPC(mpc_cfg, self.burning)

        # ======================================================
        # TIME CONFIG
        # ======================================================
        self.time = 0.0
        self.dt = cfg["simulation"]["dt"]
        self.total_hours = cfg["simulation"]["total_hours"]
        self.chunk_hours = cfg["simulation"]["chunk_hours"]

        # ======================================================
        # LOGGING (SAFE)
        # ======================================================
        self.log_interval = cfg.get("logging", {}).get("interval_sec", 60)
        self._next_log_time = 0.0

        # ======================================================
        # OPERATIONAL LAYER (NOW ACTIVE)
        # ======================================================
        self.operational = cfg.get("operational", {})

        # ======================================================
        # FUEL CONFIG (CONSISTENT MASS BASIS)
        # ======================================================
        fuel = cfg.get("fuel", {})

        petcoke = fuel.get("Petcoke_ratio", 0.0)
        rdf = fuel.get("RDF_ratio", 0.0)
        h2 = fuel.get("H2_ratio", 0.0)

        coal = 1.0 - (petcoke + rdf + h2)

        if coal < -1e-9:
            raise ValueError("Fuel ratios invalid (sum > 1.0)")

        self._last_inputs = {
            "Fuel_rate_total": fuel.get("Fuel_rate_total", 0.0),
            "Petcoke_ratio": petcoke,
            "Coal_ratio": coal,
            "RDF_ratio": rdf,
            "H2_ratio": h2,
            "O2": fuel.get("O2", 3.5),
        }

        # ======================================================
        # OPTIONAL FEED (IMPORTANT FOR m_dot_s PIPELINE)
        # ======================================================
        feed = cfg.get("feed", {})
        self._last_inputs["Feed_rate_kg_s"] = feed.get("Feed_rate_kg_s", 0.0)
        
        
    # ==========================================================
    # CENTRAL ENERGY VALIDATION
    # ==========================================================
    def _validate_energy_balances(self):

        # ======================================================
        # MAIN EQUIPMENT
        # ======================================================
        equipment_models = [
            ("Burning", self.burning),
            ("Transition", self.transition),
            ("Calciner", self.calciner),
        ]

        for equipment_name, equipment in equipment_models:

            result = validate_energy(
                energy_in=equipment.energy_in,
                energy_out=equipment.energy_out,
            )

            report_validation(
                result,
                equipment=equipment_name,
                balance_type="energy",
            )

        # ======================================================
        # PREHEATER STAGES
        # ======================================================
        for stage in self.preheater.stages:

            result = validate_energy(
                energy_in=stage.energy_in,
                energy_out=stage.energy_out,
            )

            report_validation(
                result,
                equipment=f"Preheater Stage {stage.stage_id}",
                balance_type="energy",
            )

    # --------------------------------------------------
    def _safe_inputs(self, raw):

        return {

            # ================= FUEL =================
            "Fuel_rate_total": raw.get(
                "Fuel_rate_total",
                self._last_inputs["Fuel_rate_total"],
            ),

            "Petcoke_ratio": raw.get(
                "Petcoke_ratio",
                self._last_inputs["Petcoke_ratio"],
            ),

            "Coal_ratio": raw.get(
                "Coal_ratio",
                self._last_inputs["Coal_ratio"],
            ),

            "RDF_ratio": raw.get(
                "RDF_ratio",
                self._last_inputs["RDF_ratio"],
            ),

            "H2_ratio": raw.get(
                "H2_ratio",
                self._last_inputs["H2_ratio"],
            ),

            "O2": raw.get(
                "O2",
                self._last_inputs["O2"],
            ),


            # ================= OPERATION =================
            "rpm": raw.get(
                "rpm",
                self.operational.get(
                    "rpm_default",
                    1.5
                ),
            ),

            "Feed_rate_kg_s": raw.get(
                "Feed_rate_kg_s",
                self._last_inputs.get(
                    "Feed_rate_kg_s",
                    0.0
                ),
            ),

        }
        
        
    # ==========================================================
    # GAS ENTHALPY FIXED-POINT COUPLING
    # ==========================================================
    def _solve_gas_coupling(self, inputs):

        max_iter = 50
        tol = 1e-3  # W

        H_burning_old = getattr(
            self.state,
            "Hgas_burning_out",
            0.0
        )

        H_transition_old = getattr(
            self.state,
            "Hgas_transition_out",
            H_burning_old
        )

        H_calciner_old = getattr(
            self.state,
            "Hgas_calciner_out",
            H_transition_old
        )

        for iteration in range(max_iter):

            # ======================================================
            # GAS ENTHALPY FIXED-POINT COUPLING
            # ======================================================
            self.state = self._solve_gas_coupling(
                inputs
            )


            return self.state

            # ==================================================
            # RELAXATION
            # ==================================================
            alpha = 0.5

            H_burning_old = (
                alpha * H_burning_new
                + (1.0 - alpha) * H_burning_old
            )

            H_transition_old = (
                alpha * H_transition_new
                + (1.0 - alpha) * H_transition_old
            )

            H_calciner_old = (
                alpha * H_calciner_new
                + (1.0 - alpha) * H_calciner_old
            )

        raise RuntimeError(
            "Gas enthalpy coupling did not converge "
            f"after {max_iter} iterations. "
            f"Residual = {err:.6e} W"
        )


    # --------------------------------------------------
    def step(self):
        
        self.state = self.mass_transport.apply(
        self.state,
        self.dt,
        )

        # ======================================================
        # MPC (DISABLED TEMPORARILY)
        # ======================================================
        try:
            # raw_inputs = self.mpc.compute_control(
            #     self.state,
            #     self._last_inputs,
            #     self.time,
            # )
            # self._last_inputs = self._safe_inputs(raw_inputs)
            pass

        except Exception as e:
            print("MPC FAILED:", repr(e))

        inputs = dict(self._last_inputs)
        
        # ======================================================
        # CENTRAL GAS MASS FLOW
        # ======================================================
        self.state.m_dot_g = gas_mass_balance(
            fuel_rate_total=inputs["Fuel_rate_total"],
            O2=inputs["O2"],
            eps=self.eps,
        )


        # ======================================================
        # GAS PROPERTIES
        # ======================================================
        inputs["rho_g"] = getattr(self.state, "rho_g", 1.2)


        inputs["rho_g"] = getattr(
            self.state,
            "rho_g",
            1.2,
        )

        # ======================================================
        # THERMAL ZONE ORDER
        # GAS FLOW: BURNING -> TRANSITION -> CALCINER
        # ======================================================

        # ------------------------------------------------------
        # 1. BURNING
        # ------------------------------------------------------
        self.state = self.burning.apply(
            self.state,
            inputs,
            self.dt,
        )

        # ------------------------------------------------------
        # 2. TRANSITION
        # ------------------------------------------------------
        self.state = self.transition.apply(
            self.state,
            self.dt,
        )

        # ------------------------------------------------------
        # 3. CALCINER
        # ------------------------------------------------------
        self.state = self.calciner.apply(
            self.state,
        )


        # ======================================================
        # PREHEATER
        # ======================================================
        self.state = self.preheater.apply(
            self.state,
            self.dt,
        )


        # ======================================================
        # COOLER
        # ======================================================
        self.state = self.cooler.apply(
            self.state,
            self.dt,
        )
        
        # ======================================================
        # CENTRAL ENERGY VALIDATION
        # ======================================================
        self._validate_energy_balances()

        # ======================================================
        # TIME UPDATE
        # ======================================================
        self.time += self.dt

        # ======================================================
        # LOGGING
        # ======================================================
        if self.time >= self._next_log_time:

            idx = self.state.Tg_burning.shape[0] // 2

            # ================= TEMPERATURE SAMPLES =================
            Tg_burn = float(self.state.Tg_burning[idx])
            Ts_burn = float(self.state.Ts_burning[idx])
            Tw_burn = float(self.state.Tw_burning[idx])

            Tg_trans = float(self.state.Tg_transition[idx])
            Ts_trans = float(self.state.Ts_transition[idx])
            Tw_trans = float(self.state.Tw_transition[idx])

            Tg_calc = float(self.state.Tg_calciner[idx])
            Ts_calc = float(self.state.Ts_calciner[idx])
            Tw_calc = float(self.state.Tw_calciner[idx])

            Tg_pre = float(self.state.Tg_preheater[idx])
            Ts_pre = float(self.state.Ts_preheater[idx])
            Tw_pre = float(self.state.Tw_preheater[idx])

            Tg_cool = float(self.state.Tg_cooler[idx])
            Ts_cool = float(self.state.Ts_cooler[idx])
            Tw_cool = float(self.state.Tw_cooler[idx])

            fuel_rate_total = inputs["Fuel_rate_total"]

            # ======================================================
            # TOTAL WALL LOSSES
            # ======================================================
            total_wall_loss = (
                self.state.Wall_loss_burning
                + self.state.Wall_loss_transition
                + self.state.Wall_loss_calciner
                + self.state.Wall_loss_preheater
                + self.state.Wall_loss_cooler
            )

            # ======================================================
            # TOTAL STORED ENERGY
            # ======================================================
            total_stored = (
                self.state.Burning_stored_energy_change
                + self.state.Transition_stored_energy_change
                + self.state.Calciner_stored_energy_change
                + self.state.Preheater_stored_energy_change
                + self.state.Cooler_stored_energy_change
            )

            # ======================================================
            # TOTAL EXHAUST ENTHALPY
            # ======================================================
            total_exhaust = (
                self.state.Hgas_cooler_out
                + self.state.Hsolid_cooler_out
            )

            # ======================================================
            # TOTAL REACTION HEAT SINK
            # ======================================================
            total_reaction = (
                getattr(self.state, "Preheater_Q_sink", 0.0)
                + getattr(self.state, "Calcination_Q_sink", 0.0)
            )

            # ======================================================
            # GLOBAL ENERGY RESIDUAL
            # ======================================================
            global_residual = (
                self.state.Q_burning
                - total_exhaust
                - total_wall_loss
                - total_stored
            )
            
            # ======================================================
            # MASS FLOW
            # ======================================================
            
            # ======================================================
            # MASS INVENTORY + GLOBAL MASS BALANCE
            # ======================================================

            # ------------------------------------------------------
            # ZONE CaCO3 INVENTORY
            # ------------------------------------------------------

            total_CaCO3 = 0.0

            for zone_name in [
                "preheater",
                "calciner",
                "transition",
                "burning",
                "cooler",
            ]:

                CaCO3 = np.sum(
                    np.maximum(
                        self.state.materials[zone_name].solids.CaCO3,
                        0.0,
                    )
                )

                total_CaCO3 += CaCO3


            # ------------------------------------------------------
            # TOTAL SOLID INVENTORY
            # ------------------------------------------------------

            total_solid_mass = 0.0

            for material in self.state.materials.values():

                for f in fields(SolidPhases):

                    values = getattr(
                        material.solids,
                        f.name,
                    )

                    total_solid_mass += np.sum(
                        np.maximum(
                            values,
                            0.0,
                        )
                    )



            # ------------------------------------------------------
            # TOTAL GAS INVENTORY
            # ------------------------------------------------------

            total_gas_mass = 0.0

            for material in self.state.materials.values():

                for f in fields(GasPhases):

                    values = getattr(
                        material.gases,
                        f.name,
                    )

                    total_gas_mass += np.sum(
                        np.maximum(
                            values,
                            0.0,
                        )
                    )


            # ======================================================
            # GLOBAL MASS BALANCE
            # ======================================================

            # ------------------------------------------------------
            # INITIAL MASS
            # ------------------------------------------------------

            if self.state.Initial_total_mass <= 0.0:
                self.state.Initial_total_mass = (
                    total_solid_mass + total_gas_mass
                )


            # ------------------------------------------------------
            # CURRENT INVENTORIES
            # ------------------------------------------------------

            self.state.Total_solid_inventory = (
                total_solid_mass
            )

            self.state.Total_gas_inventory = (
                total_gas_mass
            )


            # ------------------------------------------------------
            # EXPECTED TOTAL MASS
            #
            # M_expected =
            # M_initial
            # + M_feed
            # - M_clinker
            # ------------------------------------------------------

            mass_expected = (
                self.state.Initial_total_mass
                + self.state.Cumulative_feed_mass
                - self.state.Cumulative_clinker_mass
            )


            # ------------------------------------------------------
            # ACTUAL TOTAL MASS
            #
            # Solid + gas
            # ------------------------------------------------------

            mass_actual = (
                total_solid_mass
                + total_gas_mass
            )


            # ------------------------------------------------------
            # RESIDUAL
            # ------------------------------------------------------

            self.state.Global_mass_balance = (
                mass_expected
                - mass_actual
            )


            # ------------------------------------------------------
            # RELATIVE RESIDUAL
            # ------------------------------------------------------

            self.state.Global_mass_balance_relative = (
                self.state.Global_mass_balance
                / max(
                    abs(mass_expected),
                    1.0e-12,
                )
            )


            self._next_log_time += self.log_interval


        return self.state

    # --------------------------------------------------
    def run(self):

        t_end = self.total_hours * 3600.0
        n_steps = int(t_end / self.dt)

        # ======================================================
        # INIT SAFETY
        # ======================================================
        self.time = 0.0
        self._next_log_time = 0.0

        print("TWIN STARTED", flush=True)

        # ======================================================
        # MAIN LOOP
        # ======================================================
        for i in range(n_steps):

            try:
                self.step()

            except Exception as e:
                print(f"[TWIN CRASH @ step {i}] -> {repr(e)}")
                break


if __name__ == "__main__":

    # ======================================================
    # CONFIG LOAD
    # ======================================================
    twin_cfg = load_cfg("configs/twin_cfg.yaml")
    mpc_cfg = load_cfg("configs/mpc_cfg.yaml")

    # ======================================================
    # STATE INIT
    # ======================================================
    state = GlobalState()

    # ======================================================
    # TWIN INIT
    # ======================================================
    twin = Twin(
        state=state,
        cfg=twin_cfg,
        mpc_cfg=mpc_cfg,
    )

    # ======================================================
    # RUN
    # ======================================================
    twin.run()
    