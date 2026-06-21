import casadi as ca
import numpy as np
import csv
import os
from engine import KilnState, StepExecutor

class MPCModel:
    def __init__(self, dt_hours=0.05):
        self.dt = dt_hours
        self.dt_sec = dt_hours * 3600.0

    def symbolic_step(self, x, u):
        fuel, feed, air, rpm, cool = u[0], u[1], u[2], u[3], u[4]
        
        # Epsilon koruması: Division by zero ve log(0) engelleme
        eps = 1e-6
        rpm_safe = ca.fmax(rpm, 0.5) 
        rpm_eff = rpm_safe / (0.26 + rpm_safe)
        transfer_factor = ca.sqrt(1.0 / (rpm_eff + eps))

        x_next = ca.MX(x)

        Cp_g_b, Cp_s_b, Cp_w_b = 1250.0, 1150.0, 1000.0
        h_gs_b = 850.0 * transfer_factor
        h_gw_b, h_ws_b = 350.0, 400.0 * transfer_factor
        A_s_b, A_w_b, A_ws_b = 25.0, 120.0, 15.0
        C_gas_t, C_sol_t, C_wall_t = 140.0 * Cp_g_b, 4500.0 * Cp_s_b, 15000.0 * Cp_w_b

        m_air_s = (air * 1.293) / 3600.0
        m_sol_s = (ca.fmax(feed, 1.0) * 1000.0) / 3600.0

        a_g = (h_gs_b * A_s_b) + (h_gw_b * A_w_b) + (m_air_s * Cp_g_b)
        b_g = (fuel * 20000.0 * 0.56) + (h_gw_b * A_w_b * x[2]) + (h_gs_b * A_s_b * x[1]) + (m_air_s * Cp_g_b * 400.0)
        Tg_next = (C_gas_t * x[0] + self.dt_sec * b_g) / (C_gas_t + self.dt_sec * a_g + eps)

        Q_w_s = h_ws_b * A_ws_b * (x[2] - x[1])
        Tw_next = x[2] + (self.dt_sec / (C_wall_t + eps)) * (h_gw_b * A_w_b * (Tg_next - x[2]) - Q_w_s)

        a_s = (h_gs_b * A_s_b) + (h_ws_b * A_ws_b) + (m_sol_s * Cp_s_b)
        b_s = (m_sol_s * Cp_s_b * 900.0) + (h_gs_b * A_s_b * Tg_next) + (h_ws_b * A_ws_b * Tw_next)
        Ts_next = (C_sol_t * x[1] + self.dt_sec * b_s) / (C_sol_t + self.dt_sec * a_s + eps)

        T_amb = 30.0
        m_air_c = (cool * 1.2) / 3600.0
        m_sol_c = (ca.fmax(feed, 1.0) * 1000.0) / 3600.0

        W_g = m_air_c * 1050.0
        W_s = m_sol_c * 1150.0
        UA_eff = (120.0 * 45.0 * W_g) / (W_g + 120.0 * 45.0 + eps)

        tau_sol = (12000.0 * 1150.0) / (W_s + UA_eff + eps)
        Ts_ss = (W_s * Ts_next + UA_eff * T_amb) / (W_s + UA_eff + eps)

        Ts_cool_next = Ts_ss + (x[3] - Ts_ss) * ca.exp(-self.dt_sec / (tau_sol + eps))

        x_next[0] = Tg_next
        x_next[1] = Ts_next
        x_next[2] = Tw_next
        x_next[3] = Ts_cool_next

        return x_next

class MPCController:
    def __init__(self, model, N=10):
        self.opti = ca.Opti()
        self.model = model
        self.N = N
        self.U = self.opti.variable(5, N)
        self.X = self.opti.variable(4, N + 1)
        self.x_init = self.opti.parameter(4)
        self.x_ref = self.opti.parameter(4)
        self.prev_u = np.array([4.0, 40.0, 45000.0, 1.5, 80000.0]) # Güvenli default

        self.opti.subject_to(self.X[:, 0] == self.x_init)
        for k in range(N):
            self.opti.subject_to(self.X[:, k + 1] == self.model.symbolic_step(self.X[:, k], self.U[:, k]))
            self.opti.subject_to(self.U[0, k] >= 1.0); self.opti.subject_to(self.U[0, k] <= 10.0)
            self.opti.subject_to(self.U[1, k] >= 20.0); self.opti.subject_to(self.U[1, k] <= 150.0)
            self.opti.subject_to(self.U[2, k] >= 20000.0); self.opti.subject_to(self.U[2, k] <= 90000.0)
            self.opti.subject_to(self.U[3, k] >= 0.5); self.opti.subject_to(self.U[3, k] <= 4.0)
            self.opti.subject_to(self.U[4, k] >= 40000.0); self.opti.subject_to(self.U[4, k] <= 100000.0)

        cost = 0
        for k in range(N):
            cost += 50.0 * ca.sumsqr(self.X[1, k + 1] - self.x_ref[1])
            cost += 0.1 * ca.sumsqr(self.U[0, k])
        self.opti.minimize(cost)
        self.opti.solver("ipopt", {"expand": True}, {"max_iter": 50, "print_level": 0})

    def solve(self, x0, target):
        self.opti.set_value(self.x_init, x0)
        self.opti.set_value(self.x_ref, target)
        try:
            sol = self.opti.solve()
            self.prev_u = sol.value(self.U[:, 0])
            return self.prev_u
        except:
            return self.prev_u # Hata durumunda son başarılı değeri dön

if __name__ == "__main__":
    executor = StepExecutor(dt=0.05)
    model = MPCModel(dt_hours=0.05)
    controller = MPCController(model, N=10)

    current_state = KilnState(
        Tg_burning=1450.0, Ts_burning=1400.0, Tw_burning=1300.0, Ts_Cooling=800.0,
        Fuel_rate=4.0, Feed_rate=40.0, Air_flow=45000.0, kiln_rpm=1.5, Cooling_air_flow=80000.0
    )

    target_temps = [1500.0, 1400.0, 1250.0, 100.0]
    
    # CSV dosyası hazırlığı
    csv_file = "control.csv"
    headers = ["Time", "Fuel", "Feed", "RPM", "Ts_burning", "Tg_burning", "Target"]
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    print("--- MPC Kontrol Döngüsü Başlıyor (CSV Kayıtlı) ---")

    for t in np.arange(0, 5.0, 0.05):
        # 1. Set-point değişikliği (1 saat civarı)
        if 0.95 < t < 1.05:
            target_temps[1] = 1450.0

        # 2. MPC Kararı
        x0 = [current_state["Tg_burning"], current_state["Ts_burning"], current_state["Tw_burning"], current_state["Ts_Cooling"]]
        u_opt = controller.solve(x0, target_temps)
        
        inputs = {"Fuel_rate": float(u_opt[0]), "Feed_rate": float(u_opt[1]), "Air_flow": float(u_opt[2]), 
                  "kiln_rpm": float(u_opt[3]), "Cooling_air_flow": float(u_opt[4])}
        
        # 3. Fizik motoru update
        current_state = executor.perform_step(current_state, t, inputs)
        
        # 4. CSV Kaydı
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{t:.2f}", f"{inputs['Fuel_rate']:.4f}", f"{inputs['Feed_rate']:.2f}", 
                             f"{inputs['kiln_rpm']:.2f}", f"{current_state['Ts_burning']:.2f}", 
                             f"{current_state['Tg_burning']:.2f}", f"{target_temps[1]:.2f}"])

        # 5. Log
        print(f"Saat: {t:4.2f}h | Yakıt: {inputs['Fuel_rate']:4.2f} | RPM: {inputs['kiln_rpm']:4.2f} | "
              f"Katı Sıc.: {current_state['Ts_burning']:6.1f}°C | Hedef: {target_temps[1]:.1f}°C")