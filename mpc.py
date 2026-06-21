import casadi as ca
import numpy as np
import csv
import os


class IntegratedMPCModel:
    def __init__(self, dt_hours=0.05):
        self.dt = dt_hours
        self.dt_sec = dt_hours * 3600.0
        self.M_safe_vals = ca.vertcat(10.0, 2.0, 0.8, 0.4)

    def symbolic_step(self, x, u):
        fuel, feed, air, rpm, cool = u[0], u[1], u[2], u[3], u[4]
        Tg_b, Ts_b, Tw_b, Ts_c = x[0], x[1], x[2], x[3]
        eps = 1e-6

        # Fiziksel parametreler
        Cp_g_b, Cp_s_b, Cp_w_b = 1250.0, 1150.0, 1000.0
        m_air_s = (air * 1.293) / 3600.0
        m_sol_s = (feed * 1000.0) / 3600.0

        Q_pool = fuel * 20000.0 * 0.56
        Tg_next = (220.0 * Cp_g_b * Tg_b + self.dt_sec * (Q_pool + 5000.0)) / (
            220.0 * Cp_g_b + self.dt_sec * 500.0 + eps
        )
        Tw_next = Tw_b + (self.dt_sec / (15000.0 * Cp_w_b)) * (
            1000.0 * (Tg_next - Tw_b)
        )
        Ts_next = (6500.0 * Cp_s_b * Ts_b + self.dt_sec * (Q_pool * 0.8)) / (
            6500.0 * Cp_s_b + self.dt_sec * 400.0 + eps
        )

        Ts_cool_next = Ts_c + (130.0 - Ts_c) * (1.0 - ca.exp(-self.dt_sec / 9700.0))

        rate = 0.05 + 0.50 * (1.0 / (1.0 + ca.exp(-0.08 * (Ts_b - 850.0))))
        M_next = x[4:8] + self.dt_sec * (
            (feed / 3600.0 * ca.vertcat(0.76, 0.15, 0.06, 0.03)) - (rate * x[4:8] / 5.0)
        )

        return ca.vertcat(Tg_next, Ts_next, Tw_next, Ts_cool_next, M_next)


class IntegratedMPCController:
    def __init__(self, model, N=10):
        self.opti = ca.Opti()
        self.model = model
        self.N = N
        self.U = self.opti.variable(5, N)
        self.X = self.opti.variable(8, N + 1)
        self.S_T = self.opti.variable(N)
        self.x_init = self.opti.parameter(8)
        self.T_target = self.opti.parameter(1)
        self.prev_u = np.array([4.5, 40.0, 45000.0, 1.5, 80000.0])

        self.opti.subject_to(self.X[:, 0] == self.x_init)

        cost = 0
        for k in range(N):
            self.opti.subject_to(
                self.X[:, k + 1] == self.model.symbolic_step(self.X[:, k], self.U[:, k])
            )
            self.opti.subject_to(self.opti.bounded(2.0, self.U[0, k], 6.0))

            cost += (
                500.0 * ca.sumsqr(self.X[1, k + 1] - self.T_target)
                + 100000.0 * self.S_T[k]
            )
            if k > 0:
                cost += 200.0 * ca.sumsqr(self.U[0, k] - self.U[0, k - 1])

        self.opti.minimize(cost)
        self.opti.solver("ipopt", {"expand": True}, {"max_iter": 100, "print_level": 0})

    def solve(self, x0, target_Ts):
        self.opti.set_value(self.x_init, x0)
        self.opti.set_value(self.T_target, target_Ts)
        try:
            sol = self.opti.solve()
            self.prev_u = sol.value(self.U[:, 0])
            return self.prev_u
        except:
            return self.prev_u


# --- ÇALIŞTIRMA ---
model = IntegratedMPCModel()
controller = IntegratedMPCController(model)
csv_file = "control_integrated.csv"
current_state = {
    "Tg_burning": 1450.0,
    "Ts_burning": 1400.0,
    "Tw_burning": 1300.0,
    "Ts_Cooling": 800.0,
    "M_CaCO3": 12.0,
    "M_SiO2": 2.5,
    "M_Al2O3": 1.0,
    "M_Fe2O3": 0.5,
}

for t in np.arange(0, 5.0, 0.05):
    target_Ts = 1450.0 if t > 1.0 else 1400.0
    x0 = np.array([current_state[k] for k in current_state])
    u_opt = controller.solve(x0, target_Ts)

    # 1. Inputs sözlüğünü burada tanımlıyoruz
    inputs = {
        "Fuel_rate": float(u_opt[0]),
        "Feed_rate": float(u_opt[1]),
        "Air_flow": float(u_opt[2]),
        "kiln_rpm": float(u_opt[3]),
        "Cooling_air_flow": float(u_opt[4]),
    }

    # 2. Model adımını hesapla
    x_next = np.array(model.symbolic_step(x0, u_opt)).flatten()

    # 3. Durum güncelleme
    keys = list(current_state.keys())
    for i, key in enumerate(keys):
        current_state[key] = x_next[i]

import os
import csv

# Dosya yolu tanımı (çalışılan dizinde oluşması için)
csv_file = "control_integrated.csv"

# Eğer eski bir dosya varsa temizlemek istersen şu satırı açabilirsin:
# if os.path.exists(csv_file): os.remove(csv_file)

print("--- NMPC Kontrol Döngüsü Başlatılıyor ---")

for t in np.arange(0, 5.0, 0.05):
    # 1. Set-point yönetimi
    target_Ts = 1450.0 if t > 1.0 else 1400.0

    # 2. Durum vektörünü hazırla
    x0 = np.array([current_state[k] for k in current_state])

    # 3. NMPC Çözümü
    u_opt = controller.solve(x0, target_Ts)

    # 4. Inputs sözlüğü (Kayıt ve print için gerekli)
    inputs = {
        "Fuel_rate": float(u_opt[0]),
        "Feed_rate": float(u_opt[1]),
        "Air_flow": float(u_opt[2]),
        "kiln_rpm": float(u_opt[3]),
        "Cooling_air_flow": float(u_opt[4]),
    }

    # 5. Model adımı (Simülasyon motoru)
    x_next = np.array(model.symbolic_step(x0, u_opt)).flatten()

    # 6. Durum güncelleme (Sıralı eşleşme)
    keys = list(current_state.keys())
    for i, key in enumerate(keys):
        current_state[key] = x_next[i]

    # 7. Kayıt Bloğu
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Time",
                    "Fuel",
                    "Feed",
                    "RPM",
                    "Ts_burning",
                    "Tg_burning",
                    "CaCO3",
                    "Target",
                ]
            )
        writer.writerow(
            [
                f"{t:.2f}",
                f"{inputs['Fuel_rate']:.4f}",
                f"{inputs['Feed_rate']:.2f}",
                f"{inputs['kiln_rpm']:.2f}",
                f"{current_state['Ts_burning']:.2f}",
                f"{current_state['Tg_burning']:.2f}",
                f"{current_state['M_CaCO3']:.2f}",
                f"{target_Ts:.2f}",
            ]
        )

    # 8. Konsol çıktısı
    print(
        f"Saat: {t:4.2f}h | Yakıt: {inputs['Fuel_rate']:5.2f} | "
        f"Ts: {current_state['Ts_burning']:6.1f}°C (Hedef: {target_Ts:.0f}) | "
        f"CaCO3: {current_state['M_CaCO3']:5.1f} T"
    )
