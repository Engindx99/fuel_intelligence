from Kiln.GlobalState import GlobalState
from Kiln.Burning import Burning
from control.mpc import MasterMPC
import numpy as np
import yaml


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_cfg("configs/twin_cfg.yaml")


class Twin:

    def __init__(self, state, cfg, mpc=None):

        self.state = state
        self.model = Burning(
            N=cfg["plant"]["N"],
            L=cfg["plant"]["length"],
        )
        self.mpc = mpc

        self.time = 0.0
        self.dt = cfg["simulation"]["dt"]

        self.debug = True
        self.verbose_mpc = False  #  MPC spam kapalı

        # ================= LOG CONTROL =================
        self.log_interval = cfg["logging"]["interval_sec"]
        self._next_log_time = 0.0

        # ================= MPC CONTROL =================
        self.mpc_interval = cfg["control"]["mpc_interval"]
        self._next_mpc_time = 0.0

        self._last_inputs = {
            "Fuel_rate": cfg["fuel"]["fuel_rate"],
            "Petcoke": cfg["fuel"]["petcoke_fraction"],
            "Lignite": 1.0
            - cfg["fuel"]["petcoke_fraction"]
            - cfg["fuel"]["rdf_fraction"],
            "RDF_Fuel": cfg["fuel"]["rdf_fraction"],
            "O2": cfg["fuel"]["oxygen"],
        }

    # --------------------------------------------------
    def _safe_inputs(self, raw):

        return {
            "Fuel_rate": raw.get("Fuel_rate", self._last_inputs["Fuel_rate"]),
            "Petcoke": raw.get("Petcoke", self._last_inputs["Petcoke"]),
            "Lignite": raw.get("Lignite", self._last_inputs["Lignite"]),
            "RDF_Fuel": raw.get("RDF_Fuel", self._last_inputs["RDF_Fuel"]),
            "O2": raw.get("O2", self._last_inputs["O2"]),
        }

    # --------------------------------------------------
    def step(self):

        # =========================
        # CONTROL (THROTTLED MPC)
        # =========================
        if self.mpc is None:
            inputs = self._last_inputs

        else:
            # MPC only every 1 second
            if self.time >= self._next_mpc_time:

                try:
                    raw_inputs = self.mpc.compute_control(self.state)
                    self._last_inputs = self._safe_inputs(raw_inputs)

                except Exception as e:
                    # FAIL SAFE (NO PRINT FLOOD)
                    print("MPC FAILED:", repr(e), flush=True)

                self._next_mpc_time += self.mpc_interval

            inputs = self._last_inputs

        # =========================
        # PHYSICS
        # =========================
        Tg, Ts, Tw = self.model.thermal_step(
            self.state.Tg_burning,
            self.state.Ts_burning,
            self.state.Tw_burning,
            inputs,
            self.dt,
        )

        self.state.Tg_burning = Tg
        self.state.Ts_burning = Ts
        self.state.Tw_burning = Tw

        # =========================
        # TIME UPDATE
        # =========================
        self.time += self.dt

        # =========================
        # LOGGING (10 MIN GLOBAL VIEW)
        # =========================
        if self.time >= self._next_log_time:

            idx = len(self.state.Tg_burning) // 2

            Tg = float(self.state.Tg_burning[idx])
            Ts = float(self.state.Ts_burning[idx])
            Tw = float(self.state.Tw_burning[idx])

            print(
                f"[REPORT 10m] t={self.time/60:.1f} min | "
                f"Tg={Tg:.2f} K | "
                f"Ts={Ts:.2f} K | "
                f"Tw={Tw:.2f} K",
                flush=True,
            )

            self._next_log_time += self.log_interval

        return self.state

    # --------------------------------------------------
    def run(self, t_end, report_every=2000):

        n_steps = int(t_end / self.dt)

        print("TWIN STARTED", flush=True)

        for i in range(n_steps):

            self.step()

            # OPTIONAL DEBUG (RARE)
            if report_every > 0 and i % report_every == 0:

                idx = len(self.state.Tg_burning) // 2

                print(
                    f"step={i:06d} | "
                    f"time={self.time/3600:.4f} h | "
                    f"Tg={self.state.Tg_burning[idx]:7.2f} K | "
                    f"Ts={self.state.Ts_burning[idx]:7.2f} K | "
                    f"Tw={self.state.Tw_burning[idx]:7.2f} K",
                    flush=True,
                )


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    twin_cfg = load_cfg("configs/twin_cfg.yaml")
    mpc_cfg = load_cfg("configs/mpc_cfg.yaml")

    print("STARTING TWIN SIMULATION", flush=True)

    state = GlobalState()
    mpc = MasterMPC(mpc_cfg)

    twin = Twin(
        state=state,
        cfg=twin_cfg,
        mpc=mpc,
    )

    twin.run(t_end=3600, report_every=2000)
