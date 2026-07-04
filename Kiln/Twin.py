from kiln.globalstate import GlobalState
from kiln.burning import Burning
from kiln.transition import Transition
from kiln.calciner import Calciner
from kiln.preheater import Preheater
from kiln.cooler import Cooler
from controls.mpc import MasterMPC

import numpy as np
import yaml


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Twin:

    def __init__(self, state, cfg, mpc_cfg):

        self.state = state

        # ======================================================
        # BURNING MODEL
        # ======================================================
        self.burning = Burning(
            N=cfg["plant"]["N"],
            L=cfg["plant"]["length"],
        )

        # ======================================================
        # TRANSITION MODEL
        # ======================================================
        self.transition = Transition(
            N=cfg["plant"]["N"],
            L=cfg["transition"]["length"],
        )

        # ======================================================
        # CALCINER MODEL
        # ======================================================
        self.calciner = Calciner(
            N=cfg["plant"]["N"],
            L=cfg["calciner"]["length"],
        )

        # ======================================================
        # PREHEATER MODEL
        # ======================================================
        self.preheater = Preheater(
            N=cfg["plant"]["N"],
            L=cfg["preheater"]["length"],
        )

        # ======================================================
        # COOLER MODEL
        # ======================================================
        self.cooler = Cooler(
            N=cfg["plant"]["N"],
            L=cfg["cooler"]["length"],
        )

        # ======================================================
        # MPC
        # ======================================================
        #self.mpc = MasterMPC(
            #mpc_cfg,
            #self.burning,
        #)

        # ======================================================
        # TIME
        # ======================================================
        self.time = 0.0
        self.dt = cfg["simulation"]["dt"]
        self.total_hours = cfg["simulation"]["total_hours"]
        self.chunk_hours = cfg["simulation"]["chunk_hours"]

        # ======================================================
        # LOGGING
        # ======================================================
        self.log_interval = cfg["logging"]["interval_sec"]
        self._next_log_time = 0.0

        # ======================================================
        # LAST VALID INPUTS
        # ======================================================
        self._last_inputs = {

            "Fuel_rate_total": cfg["fuel"]["Fuel_rate_total"],

            "Petcoke_ratio": cfg["fuel"]["Petcoke_ratio"],

            "Coal_ratio":
                1.0
                - cfg["fuel"]["Petcoke_ratio"]
                - cfg["fuel"]["RDF_ratio"]
                - cfg["fuel"].get("H2_ratio", 0.0),

            "RDF_ratio": cfg["fuel"]["RDF_ratio"],

            "H2_ratio": cfg["fuel"].get("H2_ratio", 0.0),

            "O2": cfg["fuel"]["O2"],
        }

    # --------------------------------------------------
    def _safe_inputs(self, raw):

        return {

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
        }

    # --------------------------------------------------
    def step(self):

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
            # fallback: keep last valid control

        inputs = self._last_inputs

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
        # TIME
        # ======================================================
        self.time += self.dt

        # ======================================================
        # LOGGING
        # ======================================================
        if self.time >= self._next_log_time:

            idx = len(self.state.Tg_burning) // 2

            # ================= BURNING =================
            Tg_burn = float(self.state.Tg_burning[idx])
            Ts_burn = float(self.state.Ts_burning[idx])
            Tw_burn = float(self.state.Tw_burning[idx])

            # ================= TRANSITION =================
            Tg_trans = float(self.state.Tg_transition[idx])
            Ts_trans = float(self.state.Ts_transition[idx])
            Tw_trans = float(self.state.Tw_transition[idx])

            # ================= CALCINER =================
            Tg_calc = float(self.state.Tg_calciner[idx])
            Ts_calc = float(self.state.Ts_calciner[idx])
            Tw_calc = float(self.state.Tw_calciner[idx])

            # ================= PREHEATER =================
            Tg_pre = float(self.state.Tg_preheater[idx])
            Ts_pre = float(self.state.Ts_preheater[idx])
            Tw_pre = float(self.state.Tw_preheater[idx])

            # ================= COOLER =================
            Tg_cool = float(self.state.Tg_cooler[idx])
            Ts_cool = float(self.state.Ts_cooler[idx])
            Tw_cool = float(self.state.Tw_cooler[idx])

            fuel_rate_total = inputs["Fuel_rate_total"]

            total_wall_loss = (
                self.state.Wall_loss_burning
                + self.state.Wall_loss_transition
                + self.state.Wall_loss_calciner
                + self.state.Wall_loss_preheater
                + self.state.Wall_loss_cooler
            )

            total_stored = (
                self.state.Burning_stored_energy_change
                + self.state.Transition_stored_energy_change
                + self.state.Calciner_stored_energy_change
                + self.state.Preheater_stored_energy_change
                + self.state.Cooler_stored_energy_change
            )

            print(
                f"\n========== DIGITAL TWIN REPORT ==========\n"

                f"Time            : {self.time/60:.1f} min\n"
                f"Fuel Rate       : {fuel_rate_total:.2f} t/h\n"

                f"\n--- Burning -----------------------------\n"
                f"Tg              : {Tg_burn:.2f} K\n"
                f"Ts              : {Ts_burn:.2f} K\n"
                f"Tw              : {Tw_burn:.2f} K\n"
                f"Q_burning       : {self.state.Q_burning/1e6:.2f} MW\n"
                f"Hgas_out        : {self.state.Hgas_burning_out/1e6:.2f} MW\n"
                f"Wall loss       : {self.state.Wall_loss_burning/1e6:.2f} MW\n"
                f"Stored          : {self.state.Burning_stored_energy_change/1e6:.2f} MW\n"
                f"Residual        : {self.state.Burning_energy_balance/1e6:.2f} MW\n"

                f"\n--- Transition --------------------------\n"
                f"Tg              : {Tg_trans:.2f} K\n"
                f"Ts              : {Ts_trans:.2f} K\n"
                f"Tw              : {Tw_trans:.2f} K\n"
                f"Hgas_in         : {self.state.Hgas_burning_out/1e6:.2f} MW\n"
                f"Hgas_out        : {self.state.Hgas_transition_out/1e6:.2f} MW\n"
                f"Wall loss       : {self.state.Wall_loss_transition/1e6:.2f} MW\n"
                f"Stored          : {self.state.Transition_stored_energy_change/1e6:.2f} MW\n"
                f"Residual        : {self.state.Transition_energy_balance/1e6:.2f} MW\n"

                f"\n--- Calciner ----------------------------\n"
                f"Tg              : {Tg_calc:.2f} K\n"
                f"Ts              : {Ts_calc:.2f} K\n"
                f"Tw              : {Tw_calc:.2f} K\n"
                f"Hgas_in         : {self.state.Hgas_transition_out/1e6:.2f} MW\n"
                f"Hgas_out        : {self.state.Hgas_calciner_out/1e6:.2f} MW\n"
                f"Calcination     : {self.state.Calciner_Q_sink/1e6:.2f} MW\n"
                f"Wall loss       : {self.state.Wall_loss_calciner/1e6:.2f} MW\n"
                f"Stored          : {self.state.Calciner_stored_energy_change/1e6:.2f} MW\n"
                f"Residual        : {self.state.Calciner_energy_balance/1e6:.2f} MW\n"

                f"\n--- Preheater ---------------------------\n"
                f"Tg              : {Tg_pre:.2f} K\n"
                f"Ts              : {Ts_pre:.2f} K\n"
                f"Tw              : {Tw_pre:.2f} K\n"
                f"Hgas_in         : {self.state.Hgas_calciner_out/1e6:.2f} MW\n"
                f"Hgas_out        : {self.state.Hgas_preheater_out/1e6:.2f} MW\n"
                f"Wall loss       : {self.state.Wall_loss_preheater/1e6:.2f} MW\n"
                f"Stored          : {self.state.Preheater_stored_energy_change/1e6:.2f} MW\n"
                f"Residual        : {self.state.Preheater_energy_balance/1e6:.2f} MW\n"

                f"\n--- Cooler ------------------------------\n"
                f"Tg              : {Tg_cool:.2f} K\n"
                f"Ts              : {Ts_cool:.2f} K\n"
                f"Tw              : {Tw_cool:.2f} K\n"
                f"Hgas_in         : {self.state.Hgas_preheater_out/1e6:.2f} MW\n"
                f"Hgas_out        : {self.state.Hgas_cooler_out/1e6:.2f} MW\n"
                f"Wall loss       : {self.state.Wall_loss_cooler/1e6:.2f} MW\n"
                f"Stored          : {self.state.Cooler_stored_energy_change/1e6:.2f} MW\n"
                f"Residual        : {self.state.Cooler_energy_balance/1e6:.2f} MW\n"

                f"\n========== GLOBAL ENERGY SUMMARY ==========\n"
                f"Fuel input      : {self.state.Q_burning/1e6:.2f} MW\n"
                f"Final exhaust   : {self.state.Hgas_cooler_out/1e6:.2f} MW\n"
                f"Wall losses     : {total_wall_loss/1e6:.2f} MW\n"
                f"Stored energy   : {total_stored/1e6:.2f} MW\n"
                f"Calcination     : {self.state.Calciner_Q_sink/1e6:.2f} MW\n"
                f"==========================================\n",

                flush=True,
            )

            self._next_log_time += self.log_interval

        return self.state

    # --------------------------------------------------
    def run(self):

        t_end = self.total_hours * 3600.0
        n_steps = int(t_end / self.dt)

        print("TWIN STARTED", flush=True)

        for _ in range(n_steps):
            self.step()

if __name__ == "__main__":

    twin_cfg = load_cfg("configs/twin_cfg.yaml")
    mpc_cfg = load_cfg("configs/mpc_cfg.yaml")

    state = GlobalState()

    twin = Twin(
        state=state,
        cfg=twin_cfg,
        mpc_cfg=mpc_cfg,
    )

    twin.run()
    