from kiln.globalstate import GlobalState
from kiln.burning import Burning
from kiln.transition import Transition
from kiln.calciner import Calciner
from kiln.preheater import Preheater
from kiln.cooler import Cooler
from controls.mpc import MasterMPC
from physics.mass_transport import MassTransport
from dataclasses import fields
from chemistry.phases import SolidPhases

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
        # MASS FLOW INJECTION
        # ======================================================
        inputs["m_dot_g"] = self.state.m_dot_g
        inputs["rho_g"] = getattr(self.state, "rho_g", 1.2)

        # ======================================================
        # BURNING
        # ======================================================
        self.state = self.burning.apply(
            self.state,
            inputs,
            self.dt,
        )


        # ======================================================
        # TRANSITION
        # ======================================================
        self.state = self.transition.apply(
            self.state,
            self.dt,
        )


        # ======================================================
        # CALCINER
        # ======================================================
        self.state = self.calciner.apply(
            self.state,
            self.dt,
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
            
            print("\n========== MASS INVENTORY (CaCO3) ==========")

            for zone in [
                "preheater",
                "calciner",
                "transition",
                "burning",
                "cooler",
            ]:
                print(
                    f"{zone:11s}: "
                    f"{np.sum(self.state.materials[zone].solids.CaCO3):10.2f} kg"
                )
                
            total_solids = 0.0

            for zone in state.materials.values():
                for f in fields(SolidPhases):
                    total_solids += np.sum(getattr(zone.solids, f.name))

            print(f"TOTAL SOLIDS = {total_solids:.2f} kg")
    
            # ======================================================
            # ZONE TEMPERATURES
            # ======================================================
            

            print("\n========== ZONE TEMPERATURES ==========")

            print(
                f"BURNING   | "
                f"Tg_mean: {np.mean(self.state.Tg_burning):.2f} K | "
                f"Ts_mean: {np.mean(self.state.Ts_burning):.2f} K | "
                f"Tw_mean: {np.mean(self.state.Tw_burning):.2f} K"
            )

            print(
                f"TRANSITION| "
                f"Tg_mean: {np.mean(self.state.Tg_transition):.2f} K | "
                f"Ts_mean: {np.mean(self.state.Ts_transition):.2f} K | "
                f"Tw_mean: {np.mean(self.state.Tw_transition):.2f} K"
            )

            print(
                f"CALCINER  | "
                f"Tg_mean: {np.mean(self.state.Tg_calciner):.2f} K | "
                f"Ts_mean: {np.mean(self.state.Ts_calciner):.2f} K | "
                f"Tw_mean: {np.mean(self.state.Tw_calciner):.2f} K"
            )

            print(
                f"PREHEATER | "
                f"Tg_mean: {np.mean(self.state.Tg_preheater):.2f} K | "
                f"Ts_mean: {np.mean(self.state.Ts_preheater):.2f} K | "
                f"Tw_mean: {np.mean(self.state.Tw_preheater):.2f} K"
            )

            print(
                f"COOLER    | "
                f"Tg_mean: {np.mean(self.state.Tg_cooler):.2f} K | "
                f"Ts_mean: {np.mean(self.state.Ts_cooler):.2f} K | "
                f"Tw_mean: {np.mean(self.state.Tw_cooler):.2f} K"
            )

            # ======================================================
            # GLOBAL ENERGY
            # ======================================================

            print("\n========== GLOBAL ENERGY ==========")

            print(f"Fuel       : {self.state.Q_burning:.2f} W")
            print(f"Exhaust    : {total_exhaust:.2f} W")
            print(f"Wall loss  : {total_wall_loss:.2f} W")
            print(f"Reaction   : {total_reaction:.2f} W")
            print(f"Stored     : {total_stored:.2f} W")
            print(f"Residual   : {global_residual:.2f} W")

            relative = (
                global_residual /
                (
                    abs(self.state.Q_burning)
                    + self.eps
                )
            )

            print(f"Relative   : {relative:.6f}")

            # ======================================================
            # REACTION BREAKDOWN
            # ======================================================

            print("\n========== REACTION BREAKDOWN ==========")

            print(
                f"Preheater  : {getattr(self.state, 'Preheater_Q_sink', 0.0):.2f} W"
            )

            print(
                f"Calciner   : {getattr(self.state, 'Calcination_Q_sink', 0.0):.2f} W"
            )

            # ======================================================
            # STORED BREAKDOWN
            # ======================================================

            print("\n========== STORED BREAKDOWN ==========")

            print(
                f"Burning    : {self.state.Burning_stored_energy_change:.2f} W"
            )

            print(
                f"Transition : {self.state.Transition_stored_energy_change:.2f} W"
            )

            print(
                f"Calciner   : {self.state.Calciner_stored_energy_change:.2f} W"
            )

            print(
                f"Preheater  : {self.state.Preheater_stored_energy_change:.2f} W"
            )

            print(
                f"Cooler     : {self.state.Cooler_stored_energy_change:.2f} W"
            )

            # ======================================================
            # ZONE ENERGY BALANCES
            # ======================================================

            print("\n========== ZONE ENERGY BALANCES ==========")

            print(
                f"Burning    : {self.state.Burning_energy_balance:.2f} W"
            )

            print(
                f"Transition : {self.state.Transition_energy_balance:.2f} W"
            )

            print(
                f"Calciner   : {self.state.Calciner_energy_balance:.2f} W"
            )

            print(
                f"Preheater  : {self.state.Preheater_energy_balance:.2f} W"
            )

            print(
                f"Cooler     : {self.state.Cooler_energy_balance:.2f} W"
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
    