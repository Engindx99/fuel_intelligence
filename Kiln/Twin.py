from Kiln.GlobalState import GlobalState
from Kiln.Burning import Burning
from control.mpc import MasterMPC
import numpy as np


class Twin:

    def __init__(self, state, mpc=None, N=5):

        self.state = state
        self.model = Burning(N=N)
        self.mpc = mpc

        self.time = 0.0
        self.dt = 0.05

        self.debug = True
        self.verbose_mpc = False  # 🔥 terminal spam engellendi

        # ================= LOG CONTROL =================
        self.log_interval = 60.0  # 1 minute
        self._next_log_time = 0.0

        # ================= MPC CONTROL =================
        self.mpc_interval = 1.0  # 🔥 MPC 1 saniyede bir çalışır
        self._next_mpc_time = 0.0

        self._last_inputs = {
            "Fuel_rate": 5.5,
            "Petcoke": 0.6,
            "Lignite": 0.2,
            "RDF_Fuel": 0.2,
            "O2": 3.5,
        }

    # --------------------------------------------------
    def _safe_inputs(self, raw):

        return {
            "Fuel_rate": raw.get("Fuel_rate", 5.5),
            "Petcoke": raw.get("Petcoke", 0.6),
            "Lignite": raw.get("Lignite", 0.2),
            "RDF_Fuel": raw.get("RDF_Fuel", 0.2),
            "O2": raw.get("O2", 3.5),
        }

    # --------------------------------------------------
    def step(self):

        # =========================
        # CONTROL (THROTTLED MPC)
        # =========================
        if self.mpc is None:

            inputs = self._last_inputs

        else:

            # 🔥 MPC only every 1 second
            if self.time >= self._next_mpc_time:

                try:
                    raw_inputs = self.mpc.compute_control(self.state)
                    self._last_inputs = self._safe_inputs(raw_inputs)

                except Exception as e:

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
        # LOGGING (1 MIN INTERVAL)
        # =========================
        if self.time >= self._next_log_time:

            idx = len(self.state.Tg_burning) // 2

            print(
                f"[t={self.time:6.1f}s] "
                f"Tg={self.state.Tg_burning[idx]:7.2f}K | "
                f"Ts={self.state.Ts_burning[idx]:7.2f}K | "
                f"Tw={self.state.Tw_burning[idx]:7.2f}K",
                flush=True,
            )

            self._next_log_time += self.log_interval

        return self.state

    # --------------------------------------------------
    def run(self, t_end, report_every=1000):

        n_steps = int(t_end / self.dt)

        print("TWIN STARTED", flush=True)

        for i in range(n_steps):

            self.step()

            # extra debug report
            if report_every > 0 and i % report_every == 0:

                idx = len(self.state.Tg_burning) // 2

                print(
                    f"step={i:06d} | "
                    f"time={self.time/3600:.4f} h | "
                    f"Tg={self.state.Tg_burning[idx]:7.2f}K | "
                    f"Ts={self.state.Ts_burning[idx]:7.2f}K | "
                    f"Tw={self.state.Tw_burning[idx]:7.2f}K",
                    flush=True,
                )


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    print("STARTING TWIN SIMULATION", flush=True)

    state = GlobalState()
    mpc = MasterMPC()

    twin = Twin(state, mpc, N=5)

    twin.run(t_end=3600, report_every=1000)
