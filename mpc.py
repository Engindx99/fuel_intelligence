import casadi as ca
import numpy as np
import csv


class IntegratedMPCModel:
    def __init__(self, dt_hours=0.05):
        self.dt = dt_hours
        self.dt_sec = dt_hours * 3600.0
        self.composition = ca.vertcat(0.76, 0.15, 0.06, 0.03)
        self.M_safe_vals = ca.vertcat(10.0, 2.0, 0.8, 0.4)
        self.rates_base = ca.vertcat(0.05, 0.01, 0.005, 0.002)

    def symbolic_step(self, x, u):
        """
        x: [Tg_b, Ts_b, Tw_b, Ts_c, M_caco3, M_sio2, M_al2o3, M_fe2o3]
        u: [fuel, feed, air, rpm, cool]
        """
        # Sembolik değişkenleri ayıkla
        fuel, feed, air, rpm, cool = u[0], u[1], u[2], u[3], u[4]
        Tg_b, Ts_b, Tw_b, Ts_c = x[0], x[1], x[2], x[3]
        eps = 1e-6

        # --- 1. BURNING ZONE (engine.py ile aynı katsayılar) ---
        Cp_g_b, Cp_s_b, Cp_w_b = 1250.0, 1150.0, 1000.0
        A_c_b, h_c_b = 13.85, 0.05
        A_s_b, A_w_b, A_ws_b = 82.0, 70.0, 57.0
        h_gs_b, h_gw_b, h_ws_b = 1450.0, 350.0, 400.0

        m_air_s = (air * 1.293) / 3600.0
        m_sol_s = (feed * 1000.0) / 3600.0

        C_gas_t, C_sol_t, C_wall_t = 220.0 * Cp_g_b, 6500.0 * Cp_s_b, 15000.0 * Cp_w_b

        # Q_burning_pool (basitleştirilmiş sembolik form)
        Q_pool = fuel * 20000.0 * 0.56

        a_g = (
            (h_gs_b * A_s_b)
            + (h_gw_b * A_w_b)
            + (h_c_b * 1000.0 * A_c_b)
            + (m_air_s * Cp_g_b)
        )
        b_g = (
            Q_pool
            + (h_gw_b * A_w_b * Tw_b)
            + (h_c_b * 1000.0 * A_c_b * 30.0)
            + (h_gs_b * A_s_b * Ts_b)
        )
        Tg_next = (C_gas_t * Tg_b + self.dt_sec * b_g) / (
            C_gas_t + self.dt_sec * a_g + eps
        )

        # Tw_next
        Tw_next = Tw_b + (self.dt_sec / C_wall_t) * (
            h_gw_b * A_w_b * (Tg_next - Tw_b)
            - 15.0 * A_w_b * (Tw_b - 25.0)
            - h_ws_b * A_ws_b * (Tw_b - Ts_b)
        )

        # Ts_next (Exo enerji dahil)
        Q_exo = 350000.0 * ca.if_else(Ts_b > 1200.0, 1.0, 0.0)
        a_s = (h_gs_b * A_s_b) + (h_ws_b * A_ws_b) + (m_sol_s * Cp_s_b)
        b_s = (
            Q_exo
            + (m_sol_s * Cp_s_b * 900.0)
            + (h_gs_b * A_s_b * Tg_next)
            + (h_ws_b * A_ws_b * Tw_next)
        )
        Ts_next = (C_sol_t * Ts_b + self.dt_sec * b_s) / (
            C_sol_t + self.dt_sec * a_s + eps
        )

        # --- 2. COOLING ZONE (engine.py'deki üstel soğuma) ---
        dt_sec = self.dt_sec
        Ts_cool_next = (
            Ts_c
            + (130.0 - Ts_c) * (1.0 - ca.exp(-dt_sec / 9700.0))
            - (0.15 * (Ts_c - 1490.0) * dt_sec / (m_sol_s * Cp_s_b + eps))
        )

        # --- 3. KÜTLE DENGESİ (Envanter) ---
        M_curr = x[4:8]
        # Basit kalsinasyon tüketimi
        rate = 0.05 + 0.50 * (1.0 / (1.0 + ca.exp(-0.08 * (Ts_b - 850.0))))
        M_next = M_curr + self.dt_sec * (
            (feed / 3600.0 * ca.vertcat(0.76, 0.15, 0.06, 0.03)) - (rate * M_curr / 5.0)
        )

        return ca.vertcat(Tg_next, Ts_next, Tw_next, Ts_cool_next, M_next)


# --- 2. ENTEGRE MPC KONTROLCÜSÜ ---
class IntegratedMPCController:
    def __init__(self, model, N=10):
        self.opti = ca.Opti()
        self.model = model
        self.N = N

        # Değişkenler
        self.U = self.opti.variable(5, N)  # Kontrol Vektörü
        self.X = self.opti.variable(8, N + 1)  # 8 Boyutlu Durum Vektörü
        self.S = self.opti.variable(4, N)  # Kütle Güvenliği için Soft Constraint Slack

        # Parametreler
        self.x_init = self.opti.parameter(8)
        self.T_target = self.opti.parameter(1)  # Sadece Hedef Ts parametre yapıldı

        # Güvenli Başlangıç Değerleri
        self.prev_u = np.array([4.0, 40.0, 45000.0, 1.5, 80000.0])

        # Başlangıç Kısıtı
        self.opti.subject_to(self.X[:, 0] == self.x_init)

        cost = 0
        for k in range(N):
            # 1. Dinamik Model (Multiple Shooting Constraint)
            self.opti.subject_to(
                self.X[:, k + 1] == self.model.symbolic_step(self.X[:, k], self.U[:, k])
            )

            # 2. Kontrol (Eyleyici) Limitleri
            self.opti.subject_to(self.opti.bounded(1.0, self.U[0, k], 10.0))  # Fuel
            self.opti.subject_to(self.opti.bounded(20.0, self.U[1, k], 150.0))  # Feed
            self.opti.subject_to(
                self.opti.bounded(20000.0, self.U[2, k], 90000.0)
            )  # Air
            self.opti.subject_to(self.opti.bounded(0.5, self.U[3, k], 4.0))  # RPM
            self.opti.subject_to(
                self.opti.bounded(40000.0, self.U[4, k], 100000.0)
            )  # Cooling

            # 3. Kütle Kısıtları (Fiziksel Alt Sınır ve Soft Limit)
            # Envanterin negatif olmasını kesin olarak engelle (Hard Constraint)
            self.opti.subject_to(self.X[4:8, k + 1] >= 0.0)
            self.opti.subject_to(self.S[:, k] >= 0.0)

            # Güvenli seviye koruması (Soft Constraint: X + Slack >= M_safe)
            self.opti.subject_to(
                self.X[4:8, k + 1] + self.S[:, k] >= self.model.M_safe_vals
            )

            # --- MALİYET FONKSİYONU (COST FUNCTION) ---
            # A) Termal Kalite: Ts'yi hedefe ulaştır
            cost += 50.0 * ca.sumsqr(self.X[1, k + 1] - self.T_target)

            # B) Kütle Güvenliği: Slack değişkenlerine ağır ceza ver (Envanterin bitmesini engeller)
            cost += 10000.0 * ca.sumsqr(self.S[:, k])

            # C) Kontrol Eforu ve Kararlılık (Delta-U Cezaları)
            cost += 0.1 * ca.sumsqr(self.U[0, k])  # Nominal fuel penalty
            if k > 0:
                cost += 5.0 * ca.sumsqr(
                    self.U[0, k] - self.U[0, k - 1]
                )  # Yakıt dalgalanma cezası
                cost += 2.0 * ca.sumsqr(
                    self.U[1, k] - self.U[1, k - 1]
                )  # Besleme dalgalanma cezası

        self.opti.minimize(cost)

        # Çözücü Ayarları (Daha sessiz ve sağlam IPopt ayarları)
        self.opti.solver(
            "ipopt",
            {"expand": True},
            {"max_iter": 100, "print_level": 0, "acceptable_tol": 1e-4},
        )

    def solve(self, x0, target_Ts):
        self.opti.set_value(self.x_init, x0)
        self.opti.set_value(self.T_target, target_Ts)

        # Warm Start
        self.opti.set_initial(self.U, np.tile(self.prev_u.reshape(-1, 1), (1, self.N)))

        try:
            sol = self.opti.solve()
            self.prev_u = sol.value(self.U[:, 0])
            return self.prev_u
        except RuntimeError:
            # Hata durumunda (Infeasible), en iyi suboptimal çözümü döndür
            return self.opti.debug.value(self.U[:, 0])


# --- SIMÜLASYON MOTORU BAŞLATICI ---
# Sınıf tanımlarının altına, döngüden hemen önce bunları ekle:

model = IntegratedMPCModel(dt_hours=0.05)
controller = IntegratedMPCController(model, N=10)
csv_file = "control_integrated.csv"

# Fizik motorunu sembolik olarak bağla
x_in = ca.MX.sym("x", 8)
u_in = ca.MX.sym("u", 5)
f_out = model.symbolic_step(x_in, u_in)
f_plant = ca.Function("plant", [x_in, u_in], [f_out])

# Başlangıç durumu (Döngüye girmeden önce tanımlı olmalı)
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
target_Ts = 1400.0

# Artık buranın altına for döngünü koyabilirsin.
# Python artık model, controller ve csv_file değişkenlerini tanıyor.

# 1. Sabit Sembolleri Tanımla (Referans oluşturma)
x_in = ca.MX.sym("x", 8)
u_in = ca.MX.sym("u", 5)

# 2. Modeli bu sembollerle bağla
f_out = model.symbolic_step(x_in, u_in)

# 3. CasADi Fonksiyonunu oluştur
f_plant = ca.Function("plant", [x_in, u_in], [f_out])

print("--- Entegre NMPC Kontrol Döngüsü Başlıyor (8 Durumlu) ---")

for t in np.arange(0, 5.0, 0.05):
    # 1. Disturbance / Set-point
    if 0.95 < t < 1.05:
        target_Ts = 1450.0

    # 2. Durum Vektörünü Hazırla
    x0 = np.array(
        [
            current_state["Tg_burning"],
            current_state["Ts_burning"],
            current_state["Tw_burning"],
            current_state["Ts_Cooling"],
            current_state["M_CaCO3"],
            current_state["M_SiO2"],
            current_state["M_Al2O3"],
            current_state["M_Fe2O3"],
        ]
    )

    # 3. NMPC Çözümü
    u_opt = controller.solve(x0, target_Ts)

    inputs = {
        "Fuel_rate": float(u_opt[0]),
        "Feed_rate": float(u_opt[1]),
        "Air_flow": float(u_opt[2]),
        "kiln_rpm": float(u_opt[3]),
        "Cooling_air_flow": float(u_opt[4]),
    }

    # 4. Fizik Motoru Güncellemesi (Artık sembolik referans hatası yok)
    x_next_num = np.array(f_plant(x0, u_opt)).flatten()

    # 5. Durumu Güncelle
    current_state = {
        "Tg_burning": x_next_num[0],
        "Ts_burning": x_next_num[1],
        "Tw_burning": x_next_num[2],
        "Ts_Cooling": x_next_num[3],
        "M_CaCO3": x_next_num[4],
        "M_SiO2": x_next_num[5],
        "M_Al2O3": x_next_num[6],
        "M_Fe2O3": x_next_num[7],
    }

    # 6. Kayıt
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
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
        print(
            f"Saat: {t:4.2f}h | Yakıt: {inputs['Fuel_rate']:5.2f} | Besleme: {inputs['Feed_rate']:6.2f} | "
            f"Ts: {current_state['Ts_burning']:6.1f}°C (Hedef: {target_Ts:.0f}) | CaCO3 Envanter: {current_state['M_CaCO3']:5.1f} T"
        )
