from Kiln.GlobalState import GlobalState
from Kiln.Burning import Burning
from control.mpc import MasterMPC
import numpy as np


class Twin:

    def __init__(self, state, mpc=None, N=80):

        self.state = state
        self.burning = Burning(N=N)

        self.mpc = mpc

        self.time = 0.0

        # debug flags
        self.debug = True
        self.verbose_mpc = True

    # --------------------------------------------------

    def step(self, dt):

        # =========================
        # CONTROL INPUT
        # =========================
        if self.mpc is None:

            if self.debug:
                print("[CONTROL] Using fallback controller", flush=True)

            inputs = {
                "Fuel_rate": 5.0,
                "Petcoke": 0.60,
                "Lignite": 0.20,
                "RDF_Fuel": 0.20,
                "O2": 3.5,
            }

        else:

            if self.verbose_mpc:
                print("CALLING MPC", flush=True)

            try:
                inputs = self.mpc.compute_control(self.state)

                if self.verbose_mpc:
                    print(
                        f"MPC OK | Fuel={inputs['Fuel_rate']:.3f} | "
                        f"O2={inputs['O2']:.3f}",
                        flush=True,
                    )

            except Exception as e:

                print("MPC FAILED:", repr(e), flush=True)

                inputs = {
                    "Fuel_rate": 5.0,
                    "Petcoke": 0.60,
                    "Lignite": 0.20,
                    "RDF_Fuel": 0.20,
                    "O2": 3.5,
                }

        # =========================
        # BEFORE STATE
        # =========================
        Tg_before = np.mean(self.state.Tg_burning)

        # NaN guard (çok kritik)
        if np.isnan(Tg_before):
            print("WARNING: NaN detected in state BEFORE step", flush=True)

        # =========================
        # PHYSICS STEP
        # =========================
        self.state = self.burning.apply(self.state, inputs, dt)

        # =========================
        # AFTER STATE
        # =========================
        Tg_after = np.mean(self.state.Tg_burning)

        if np.isnan(Tg_after):
            print("WARNING: NaN detected in state AFTER step", flush=True)

        # =========================
        # EFFECT CHECK
        # =========================
        if self.debug:
            print(
                f"[STATE] Tg: {Tg_before:.2f} → {Tg_after:.2f} | "
                f"Δ={Tg_after - Tg_before:.5f}",
                flush=True,
            )

        self.time += dt

        return self.state

    # --------------------------------------------------

    def run(self, t_end, dt, report_every=100):

        n_steps = int(t_end / dt)

        print("TWIN STARTED", flush=True)

        for step in range(n_steps):

            self.step(dt)

            if step % report_every == 0:

                idx = len(self.state.Tg_burning) // 2

                print(
                    f"time={self.time:8.1f}s | "
                    f"Tg={self.state.Tg_burning[idx]-273.15:7.2f}°C | "
                    f"Ts={self.state.Ts_burning[idx]-273.15:7.2f}°C | "
                    f"Tw={self.state.Tw_burning[idx]-273.15:7.2f}°C",
                    flush=True,
                )


if __name__ == "__main__":

    print("STARTING TWIN SIMULATION", flush=True)

    state = GlobalState()
    mpc = MasterMPC()

    twin = Twin(state, mpc)

    twin.run(t_end=100, dt=0.1)
