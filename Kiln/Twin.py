from Kiln.GlobalState import GlobalState
from Kiln.Burning import Burning
from control.mpc import MasterMPC
import numpy as np


class Twin:

    def __init__(self, state, mpc=None, N=5):

        self.state = state

        # 🔥 SINGLE SOURCE OF TRUTH (physics)
        self.model = Burning(N=N)

        self.mpc = mpc

        self.time = 0.0
        self.dt = 0.05

        self.debug = True
        self.verbose_mpc = True

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
        # CONTROL (MPC OR FALLBACK)
        # =========================
        if self.mpc is None:

            inputs = {
                "Fuel_rate": 5.5,
                "Petcoke": 0.6,
                "Lignite": 0.2,
                "RDF_Fuel": 0.2,
                "O2": 3.5,
            }

        else:

            if self.verbose_mpc:
                print("CALLING MPC", flush=True)

            try:
                raw_inputs = self.mpc.compute_control(self.state)
                inputs = self._safe_inputs(raw_inputs)

                if self.verbose_mpc:
                    print(
                        f"MPC OK | Fuel={inputs['Fuel_rate']:.3f} | O2={inputs['O2']:.3f}",
                        flush=True,
                    )

            except Exception as e:

                print("MPC FAILED:", repr(e), flush=True)

                inputs = {
                    "Fuel_rate": 5.5,
                    "Petcoke": 0.6,
                    "Lignite": 0.2,
                    "RDF_Fuel": 0.2,
                    "O2": 3.5,
                }

        # =========================
        # PHYSICS (EXACT MATCH WITH TEST RUN)
        # =========================
        self.state.Tg_burning, self.state.Ts_burning, self.state.Tw_burning = (
            self.model.thermal_step(
                self.state.Tg_burning,
                self.state.Ts_burning,
                self.state.Tw_burning,
                inputs,
                self.dt,
            )
        )

        # =========================
        # LOG (IDENTICAL STYLE)
        # =========================
        idx = len(self.state.Tg_burning) // 2

        if self.debug:
            print(
                f"[step] time={self.time:6.2f}s | "
                f"Tg={self.state.Tg_burning[idx]-273.15:7.2f}°C | "
                f"Ts={self.state.Ts_burning[idx]-273.15:7.2f}°C | "
                f"Tw={self.state.Tw_burning[idx]-273.15:7.2f}°C",
                flush=True,
            )

        self.time += self.dt

        return self.state

    # --------------------------------------------------
    def run(self, t_end, report_every=1000):

        n_steps = int(t_end / self.dt)

        print("TWIN STARTED", flush=True)

        for i in range(n_steps):

            self.step()

            if i % report_every == 0:

                idx = len(self.state.Tg_burning) // 2

                print(
                    f"step={i:06d} | "
                    f"time={self.time/3600:.4f} h | "
                    f"Tg={self.state.Tg_burning[idx]-273.15:7.2f} °C | "
                    f"Ts={self.state.Ts_burning[idx]-273.15:7.2f} °C | "
                    f"Tw={self.state.Tw_burning[idx]-273.15:7.2f} °C",
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
