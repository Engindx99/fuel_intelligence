from bdb import effective
import json
import pickle
import os
import numpy as np


class Burning:

    def __init__(self, N=5, L=60.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # ================= GEOMETRY =================
        self.D = 4.2
        self.A_cross = np.pi * self.D**2 / 4.0
        self.V_total = self.A_cross * self.L
        self.V_cell = self.V_total / self.N

        # ================= INTERFACIAL AREA =================
        self.epsilon_bed = 0.35

        # fiziksel taban
        a_gs_base = 6.0 * (1.0 - self.epsilon_bed) / self.D

        # tuning factor (çok kritik: 1.0 fiziksel referans, >1 hızlandırır)
        self.k_interfacial = 1.0

        self.a_gs = self.k_interfacial * a_gs_base
        self.a_gw = 2.0 * np.pi * self.D
        self.a_ws = 0.6 * self.a_gs

        # ================= WALL =================
        self.A_wall = self.a_gw * self.L
        self.h_ext = 12.0

        self.V_wall = self.A_wall * 0.05

        # ================= PROPERTIES =================
        self.rho_g = 4.1
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= VELOCITIES =================
        self.u_g = 1.4
        self.u_s = 0.005

        # ================= HEAT TRANSFER =================
        self.hv_gs = 1300.0
        self.hv_gw = 250.0
        self.hv_ws = 300.0
        
        self.T_amb = 300.0

        # ================= FUEL =================
        self.LHV_petcoke = 32e6
        self.LHV_coal = 18e6
        self.LHV_RDF = 20e6
        self.LHV_H2 = 120e6

        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        self.eps = 1e-9

        # ================= PERFORMANCE BUFFERS =================
        self._dTg_dz = np.zeros(N)
        self._dTs_dz = np.zeros(N)

        # ================= CACHE CONSTANTS =================
        self._rho_s_Cp_s = self.rho_s * self.Cp_s
        self._rho_g_Vcell_Cp_g = self.rho_g * self.V_cell * self.Cp_g
        self._rho_wall_Vwall_Cp = self.rho_wall * self.V_wall * self.Cp_wall

    # ======================================================
    def combustion_efficiency(self, O2):
        return np.exp(-((O2 - self.O2_opt) ** 2) / self.O2_sigma2)

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt, calcination_sink=0.0):

        # ================= GRADIENTS (NO ALLOC) =================
        dTg_dz = self._dTg_dz
        dTs_dz = self._dTs_dz

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]
        
        #=============== INPUTS ==================

        fuel_rate_total = inputs.get("Fuel_rate_total", 1.0)  # ton/h
        O2 = inputs.get("O2", 3.5)
        
        # ================= FUEL MIX =================

        p = inputs.get("Petcoke_ratio", 0.50)
        c = inputs.get("Coal_ratio", 0.30)
        r = inputs.get("RDF_ratio", 0.15)
        h = inputs.get("H2_ratio", 0.05)

        norm = p + c + r + h + self.eps

        p /= norm
        c /= norm
        r /= norm
        h /= norm
    
        # ======= CONVERSION (Fuel_rate : ton/h to kg/s) =======

        fuel_rate_kg_s = fuel_rate_total * 1000.0 / 3600.0
        
        # ================= COMBUSTION =================

        eta = self.combustion_efficiency(O2)

        Q_petcoke = fuel_rate_kg_s * p * self.LHV_petcoke
        Q_coal    = fuel_rate_kg_s * c * self.LHV_coal
        Q_RDF     = fuel_rate_kg_s * r * self.LHV_RDF
        Q_H2      = fuel_rate_kg_s * h * self.LHV_H2

        Q_burning = eta * (Q_petcoke + Q_coal + Q_RDF + Q_H2)
        
            
        # ================= HEAT SOURCE: W (= J/s) =================
        
        q_vol = Q_burning / (self.V_total + self.eps)

        sink_density = calcination_sink / (self.V_total + self.eps)
        q_vol = q_vol - 0.05 * sink_density

        # ================= HEAT TRANSFER =================
        q_gs = (self.hv_gs * self.a_gs * (Tg - Ts)) / self.V_cell
        q_gw = (self.hv_gw * self.a_gw * (Tg - Tw)) / self.V_cell
        q_ws = (self.hv_ws * self.a_ws * (Ts - Tw)) / self.V_cell

        # ================= SOLID CAPACITY =================

        effective = 0.01
        C_s = self._rho_s_Cp_s
        effective_C_s = effective * C_s

        # ================= GAS CAPACITY =================

        C_g = self._rho_g_Vcell_Cp_g

        # ================= WALL CAPACITY (CACHED) =================

        C_w = self._rho_wall_Vwall_Cp

        q_loss = (self.h_ext * self.A_wall * (Tw - self.T_amb)) / (
            self.V_cell + self.eps
        )

        # ================= DYNAMICS =================
        Tg_n = Tg + dt * (-self.u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g)
        Ts_n = Ts + dt * (-self.u_s * dTs_dz + (q_gs - q_ws) / effective_C_s)
        Tw_n = Tw + dt * ((q_gw + q_ws - q_loss) / C_w)

        return (Tg_n,Ts_n,Tw_n,Q_petcoke,Q_coal,Q_RDF,Q_H2, Q_burning,)
    

    # ======================================================
    # STATE UPDATE
    # ======================================================
    def apply(self, state, inputs, dt):

        (
            Tg,
            Ts,
            Tw,
            Q_petcoke,
            Q_coal,
            Q_RDF,
            Q_H2,
            Q_burning,
        ) = self.thermal_step(
            state.Tg_burning,
            state.Ts_burning,
            state.Tw_burning,
            inputs,
            dt,
            calcination_sink=getattr(state, "Calcination_Q_sink", 0.0),
        )

        # ================= UPDATE TEMPERATURE FIELDS =================
        state.Tg_burning = Tg
        state.Ts_burning = Ts
        state.Tw_burning = Tw

        # ================= FUEL ENERGY =================
        state.Q_petcoke = Q_petcoke
        state.Q_coal = Q_coal
        state.Q_RDF = Q_RDF
        state.Q_H2 = Q_H2

        # ================= TOTAL BURNING ENERGY =================
        state.Q_burning = Q_burning

        # ================= ENERGY TO CALCINER =================
        state.Hgas_burning_out = self.gas_enthalpy_out(state.Tg_burning)

        return state
    
    # ======================================================
    def gas_enthalpy_out(self, Tg):

        m_dot_g = self.rho_g * self.u_g * self.A_cross

        H_out = m_dot_g * self.Cp_g * Tg[-1]

        return H_out
        


if __name__ == "__main__":

    # ======================================================
    # CHECKPOINT HELPERS
    # ======================================================
    def save_checkpoint(path, state):
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_checkpoint(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ======================================================
    # MODEL
    # ======================================================
    model = Burning(N=5)

    inputs = {
        "Fuel_rate_total": 5.0,  # ton/h
        "Petcoke_ratio": 0.6,
        "RDF_ratio": 0.2,
        "O2": 3.5,
    }

    dt = 0.05

    # ======================================================
    # CHUNK CONFIG
    # ======================================================
    chunk_hours = 1.0
    chunk_time = chunk_hours * 3600
    n_steps_chunk = int(chunk_time / dt)

    total_hours = 6.0
    n_chunks = int(total_hours / chunk_hours)

    ckpt_file = "burning_ckpt.pkl"
    status_file = "burning_status.jsonl"

    # ======================================================
    # LOAD OR INIT STATE
    # ======================================================
    if os.path.exists(ckpt_file):
        state = load_checkpoint(ckpt_file)
        Tg, Ts, Tw = state["Tg"], state["Ts"], state["Tw"]
        start_chunk = int(state["t"] // chunk_time)
        print(f"Resuming from chunk {start_chunk}")
    else:
        Tg = np.ones(5) * 1773.15
        Ts = np.ones(5) * 1673.15
        Tw = np.ones(5) * 873.15
        start_chunk = 0

    idx = 5 // 2

    # ======================================================
    # CHUNK LOOP
    # ======================================================
    for chunk in range(start_chunk, n_chunks):

        t_local = 0.0
        next_log_time = 0.0

        for i in range(n_steps_chunk):

            Tg, Ts, Tw = model.thermal_step(Tg, Ts, Tw, inputs, dt)
            t_local += dt

            # ==================================================
            # 10 MIN LOG (SCALAR OUTLET ONLY)
            # ==================================================
            if t_local >= next_log_time:

                log_entry = {
                    "chunk": chunk,
                    "time_h": f"{(chunk * chunk_time + t_local) / 3600.0:.4f}",
                    # ONLY OUTLET VALUES (1 SCALAR EACH)
                    "Tg": float(Tg[-1]),
                    "Ts": float(Ts[-1]),
                    "Tw": float(Tw[-1]),
                }

                with open(status_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                next_log_time += 600.0  # 10 min

        # ======================================================
        # CHECKPOINT SAVE (chunk end)
        # ======================================================
        global_time = (chunk + 1) * chunk_time

        state = {"Tg": Tg, "Ts": Ts, "Tw": Tw, "t": global_time}

        save_checkpoint(ckpt_file, state)

        # ======================================================
        # PRINT LOG
        # ======================================================
        print(
            f"[CHUNK {chunk}] "
            f"t={global_time/3600:.2f} h | "
            f"Tg={Tg[-1]:.2f} K | "
            f"Ts={Ts[-1]:.2f} K | "
            f"Tw={Tw[-1]:.2f} K"
        )
