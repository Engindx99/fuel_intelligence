from kiln.globalstate import GlobalState
from kiln.burning import Burning
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
        # MPC
        # ======================================================
        self.mpc = MasterMPC(
            mpc_cfg,
            self.burning,
        )

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

            "Fuel_rate": cfg["fuel"]["fuel_rate"],

            "Petcoke": cfg["fuel"]["petcoke_fraction"],

            "Lignite":
                1.0
                - cfg["fuel"]["petcoke_fraction"]
                - cfg["fuel"]["rdf_fraction"],

            "RDF_Fuel": cfg["fuel"]["rdf_fraction"],

            "O2": cfg["fuel"]["oxygen"],
        }

    # --------------------------------------------------
    def _safe_inputs(self, raw):

        return {

            "Fuel_rate":
                raw.get(
                    "Fuel_rate",
                    self._last_inputs["Fuel_rate"],
                ),

            "Petcoke":
                raw.get(
                    "Petcoke",
                    self._last_inputs["Petcoke"],
                ),

            "Lignite":
                raw.get(
                    "Lignite",
                    self._last_inputs["Lignite"],
                ),

            "RDF_Fuel":
                raw.get(
                    "RDF_Fuel",
                    self._last_inputs["RDF_Fuel"],
                ),

            "O2":
                raw.get(
                    "O2",
                    self._last_inputs["O2"],
                ),
        }

    # --------------------------------------------------
    def step(self):

        # ======================================================
        # MPC (scheduled inside MPC)
        # ======================================================
        try:
            raw_inputs = self.mpc.compute_control(
                self.state,
                self._last_inputs,
                self.time,
            )

            self._last_inputs = self._safe_inputs(raw_inputs)

        except Exception as e:
            print("MPC FAILED:", repr(e))
            # fallback: keep last valid control

        inputs = self._last_inputs

        # ======================================================
        # PHYSICS ONLY
        # ======================================================
        Tg, Ts, Tw = self.burning.thermal_step(
            self.state.Tg_burning,
            self.state.Ts_burning,
            self.state.Tw_burning,
            inputs,
            self.dt,
        )

        self.state.Tg_burning = Tg
        self.state.Ts_burning = Ts
        self.state.Tw_burning = Tw

        # ======================================================
        # TIME
        # ======================================================
        self.time += self.dt

        # ======================================================
        # LOGGING
        # ======================================================
        if self.time >= self._next_log_time:

            idx = len(self.state.Tg_burning) // 2

            Tg = float(self.state.Tg_burning[idx])
            Ts = float(self.state.Ts_burning[idx])
            Tw = float(self.state.Tw_burning[idx])

            fuel_rate = inputs["Fuel_rate"]

            # --------------------------------------------------
            # MPC reference tracking
            # --------------------------------------------------
            Tg_ref = self.mpc.cfg["mpc"]["Tg_setpoint"]
            Ts_ref = self.mpc.cfg["mpc"]["Ts_setpoint"]

            eTg = Tg_ref - Tg
            eTs = Ts_ref - Ts

            print(
                f"[REPORT] "
                f"t={self.time/60:.1f} min | "
                f"Fuel={fuel_rate:.3f} t/h | "
                f"Tg={Tg:.2f} K | eTg={eTg:+.2f} K | "
                f"Ts={Ts:.2f} K | eTs={eTs:+.2f} K | "
                f"Tw={Tw:.2f} K",
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
    