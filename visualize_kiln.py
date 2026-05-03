"""
Fuel Intelligence - Rotary Kiln Visualization Tool (18-State Optimized)

Hızlandırma (özet): maliyet ~ adım_sayısı × N_hücre; adımlar motor `sim.t` ile biter (CFL `dt` küçültürse
sabit `sim_hours/dt` döngüsü hem yanlış süre hem ek yük üretebilir). Ayrıntılı seçenekler: `--help`.

Her koşuda `core.physics` / `kinetics` / `flow` / `simulation.engine` yeniden yüklenir; böylece
`model_config.yaml` değişince aynı IDE sürecinde bile çıktı güncellenir (önceden import’ta kilitlenen sabitler).
"""

import argparse
import hashlib
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from core.state import *


def _reload_motor_config_modules() -> None:
    """model_config.yaml physics/kinetics/flow içinde import anında sabitlendiği için
    aynı süreçte tekrar çalıştırmada eski değerler kullanılırdı. Her koşudan önce tazele."""
    import importlib
    import sys

    for name in ("core.kinetics", "core.physics", "core.flow", "simulation.engine"):
        importlib.import_module(name)
        importlib.reload(sys.modules[name])


def run_and_visualize(
    *,
    L: float = 60.0,
    n_cells: int = 100,
    dt: float = 0.05,
    sim_hours: float = 2.0,
    max_sim_hours: float = 2.0,
    log_interval: int = 1000,
    trend_sample_min: float = 3.0,
):
    # --- 1. SİMÜLASYON AYARLARI ---
    sim_hours = float(min(float(sim_hours), float(max_sim_hours)))
    t_end_s = sim_hours * 3600.0
    dt_init = float(dt)

    _root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(_root, "configs", "model_config.yaml")

    _reload_motor_config_modules()
    import core.physics as _physics
    import core.flow as _flow
    import simulation.engine as _engine

    with open(cfg_path, "rb") as cf:
        cfg_bytes = cf.read()
    cfg_yaml = yaml.safe_load(cfg_bytes.decode("utf-8")) or {}
    nominal_fuel = float(
        ((cfg_yaml.get("thermal") or {}).get("calibration_reference") or {}).get(
            "nominal_fuel_kg_s", 16.0
        )
    )

    print(
        f"[visualize_kiln] Motor YAML tazelendi | burner_eff={getattr(_physics, '_burner_eff', '?')} | "
        f"fuel->solid_frac={getattr(_physics, '_fuel_direct_solid_frac', '?')}",
        flush=True,
    )

    sim = _engine.KilnSimulation(L, n_cells, dt_init)

    # Sabit u: yakıt nominal (16 kg/s sınıfı 1750 K bandı için yeterli; plato ısı taşınımı/denge ile ilgilidir).
    u = create_zero_control()
    u[IDX_FUEL] = nominal_fuel
    u[IDX_FAN] = 800.0  # m3/h → v_gas_model mdot
    u[IDX_FEED] = 10.0  # kg/s → load_factor + inlet enthalpy scale (engine)
    u[IDX_REACTOR] = 2.0  # RPM ↑ → v_s ↑ → daha kısa kalış, eksende daha az ısınma

    initial_x = create_zero_state(n_cells)
    zs = np.linspace(0.0, L, n_cells)
    initial_x[:, IDX_T_S] = sim._T_s_in
    tg_ref = sim.gas_ic_axial_k(u, zs / L, t=0.0)
    ic_tg_blend = 0.40
    initial_x[:, IDX_T_G] = sim._T_s_in + ic_tg_blend * (tg_ref - sim._T_s_in)
    initial_x[:, IDX_CaCO3] = 0.8
    initial_x[:, IDX_SiO2] = 0.14
    initial_x[:, IDX_Al2O3] = 0.04
    initial_x[:, IDX_Fe2O3] = 0.02
    initial_x[:, IDX_EPSILON] = 0.4
    sim.set_initial_condition(initial_x)

    v_s, v_g = _flow.compute_velocities(sim._xc_for_velocities(sim.X), u)
    dz = L / n_cells
    vm = max(abs(v_s), abs(v_g))
    cfl0 = vm * dt_init / (dz + 1e-18)
    if cfl0 > 0.75:
        dt_suggest = 0.75 * dz / (vm + 1e-12)
        print(
            f"[UYARI] Başlangıç CFL≈{cfl0:.2f} (>0.75): motor dt küçültebilir. "
            f"Daha hızlı koşu için --dt {dt_suggest:.4f} veya --cells artırın.",
            flush=True,
        )

    # Trend: sabit simülasyon zamanı aralığında (ilk nokta t=0 zaten kayıtlı).
    period_s = max(float(trend_sample_min) * 60.0, dt_init)
    next_sample_t = period_s
    history_t = [0.0]
    history_tg_zL = [float(sim.X[-1, IDX_T_G])]
    history_ts_zL = [float(sim.X[-1, IDX_T_S])]
    # Δ(Tg-Ts) diagnostics
    z_frac = np.linspace(0.0, 1.0, n_cells)
    hot_mask = z_frac >= 0.80
    history_dT_zL = [float(sim.X[-1, IDX_T_G] - sim.X[-1, IDX_T_S])]
    history_dT_hot_mean = [float(np.mean(sim.X[hot_mask, IDX_T_G] - sim.X[hot_mask, IDX_T_S]))]

    print(
        f"Simülasyon: {sim_hours:.2f} saate kadar (t_end={t_end_s:.0f} s), N={n_cells}, dt0={dt_init} s.",
        flush=True,
    )
    start_time = time.time()
    step_idx = 0
    current_X = sim.X

    try:
        while sim.t < t_end_s:
            current_X = sim.step(u)
            step_idx += 1
            while next_sample_t <= sim.t:
                history_t.append(next_sample_t / 3600.0)
                history_tg_zL.append(float(current_X[-1, IDX_T_G]))
                history_ts_zL.append(float(current_X[-1, IDX_T_S]))
                history_dT_zL.append(float(current_X[-1, IDX_T_G] - current_X[-1, IDX_T_S]))
                history_dT_hot_mean.append(
                    float(np.mean(current_X[hot_mask, IDX_T_G] - current_X[hot_mask, IDX_T_S]))
                )
                next_sample_t += period_s

            if log_interval and step_idx % log_interval == 0:
                prog = min(100.0, 100.0 * sim.t / t_end_s)
                print(f"İlerleme: %{prog:.1f} | t: {sim.t/3600:.2f} sa | adım: {step_idx}", end="\r", flush=True)

    except Exception as e:
        print(f"\n[HATA] Simülasyon sırasında hata oluştu: {e}")
        return

    wall = time.time() - start_time
    print(
        f"\nSimülasyon bitti. sim.t={sim.t/3600:.3f} sa, son dt={sim.dt:.5f} s, "
        f"adım={step_idx}, süre={wall:.1f} s ({step_idx/(wall+1e-9):.0f} adım/s).",
        flush=True,
    )
    print(
        f"Final: Ts@L={float(sim.X[-1, IDX_T_S]):.1f} K | Tg@L={float(sim.X[-1, IDX_T_G]):.1f} K | "
        f"dT@L={float(sim.X[-1, IDX_T_G]-sim.X[-1, IDX_T_S]):.1f} K | "
        f"dT_hot_mean(z/L>=0.8)={float(np.mean(sim.X[hot_mask, IDX_T_G]-sim.X[hot_mask, IDX_T_S])):.1f} K",
        flush=True,
    )

    cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()[:16]
    stamp = (
        f"Fuel Intelligence | sim_hours={sim_hours} | N_CELLS={n_cells} | "
        f"dt_init={dt_init} | dt_last={sim.dt} | steps={step_idx} | "
        f"fuel_kg_s={nominal_fuel} | "
        f"model_config.sha256[{cfg_sha}] | "
        f"fuel_mode={cfg_yaml.get('thermal', {}).get('fuel_heat_mode', '?')} | "
        f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} local"
    )
    manifest = stamp + os.linesep + (
        "Outputs: kiln_temp_profile, kiln_temp_trend, kiln_chem_*, kiln_clinker_quality.png"
        + os.linesep
    )
    with open(
        os.path.join(_root, "plots_run_manifest.txt"), "w", encoding="utf-8"
    ) as mf:
        mf.write(manifest)

    # --- 2. GÖRSELLEŞTİRME ---
    sns.set_theme(style="darkgrid")
    z = np.linspace(0, L, n_cells)
    
    # Plot 1: Sıcaklık Profili
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_T_G], 'r-', label='Gaz Sıcaklığı (Tg)', linewidth=1)
    plt.plot(z, current_X[:, IDX_T_S], 'b-', label='Katı Sıcaklığı (Ts)', linewidth=1)
    plt.title("Fırın Boyunca Sıcaklık Profili (Steady-State)", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Sıcaklık [K]")
    plt.legend()
    plt.savefig("kiln_temp_profile.png", dpi=200)
    plt.close()
    
    # Plot 2: Ana Fazlar
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_CaCO3], label='CaCO3', linewidth=1)
    plt.plot(z, current_X[:, IDX_CaO], label='CaO', linewidth=1)
    plt.plot(z, current_X[:, IDX_C2S], label='C2S (Belit)', linewidth=1)
    plt.plot(z, current_X[:, IDX_C3S], label='C3S (Alit)', linewidth=1, color='darkgreen')
    plt.title("Ana Kimyasal Faz Dönüşümü", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Kütle Oranı")
    plt.legend()
    plt.savefig("kiln_chem_main.png", dpi=200)
    plt.close()

    # Plot 3: Oksitler ve Azınlık Fazlar
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_SiO2], '--', label='SiO2')
    plt.plot(z, current_X[:, IDX_Al2O3], ':', label='Al2O3')
    plt.plot(z, current_X[:, IDX_Fe2O3], '-.', label='Fe2O3')
    plt.plot(z, current_X[:, IDX_C3A], label='C3A', color='orange')
    plt.plot(z, current_X[:, IDX_C4AF], label='C4AF', color='brown')
    plt.title("Oksitler ve Yan Fazlar", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Kütle Oranı")
    plt.legend()
    plt.savefig("kiln_chem_minor.png", dpi=200)
    plt.close()
    
    # Plot 4: tek eksen — yalnızca z=L: Tg ve Ts (sim.X[-1, …]).
    plt.figure(figsize=(10, 6))
    plt.plot(history_t, history_tg_zL, "r-", linewidth=1.2, label="Tg @ z=L")
    plt.plot(history_t, history_ts_zL, "b-", linewidth=1.2, label="Ts @ z=L")
    plt.plot(history_t, history_dT_hot_mean, "k--", linewidth=1.0, label="dT(Tg-Ts) mean (z/L>=0.8)")
    plt.title(
        f"Sıcaklık trendi — Tg ve Ts @ z=L (sim.X, ~{trend_sample_min:.0f} dk örnekleme)",
        fontsize=12,
    )
    plt.xlabel("Zaman [saat]")
    plt.ylabel("Sıcaklık [K]")
    plt.xlim(0.0, float(sim_hours))
    plt.legend(loc="best")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("kiln_temp_trend.png", dpi=200)
    plt.close()
    
    # Plot 5: Klinkerleşme Oranı (Litre Ağırlığı/Kalite Göstergesi)
    plt.figure(figsize=(10, 6))
    # SOLID_SPECIES içindeki tüm bileşenlerin toplamına bölerek normalize et
    total_solid = np.sum(current_X[:, SOLID_SPECIES], axis=1)
    clinker_quality = np.divide(current_X[:, IDX_C3S], total_solid, 
                                out=np.zeros_like(total_solid), 
                                where=total_solid!=0)
    
    plt.fill_between(z, clinker_quality, color='green', alpha=0.3, label='C3S / Toplam Katı')
    plt.title("Klinkerleşme İlerlemesi (C3S Formasyonu)", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Bağıl Oran")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig("kiln_clinker_quality.png", dpi=200)
    plt.close()

    print("Analiz grafikleri (PNG) başarıyla güncellendi.", flush=True)
    print(f"Çalıştırma manifesti yazıldı: plots_run_manifest.txt", flush=True)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Kiln simülasyonu + PNG. Hız: --preset fast veya --cells / --dt / --hours."
    )
    p.add_argument(
        "--preset",
        choices=("quality", "fast"),
        default="quality",
        help="quality: N=100, dt=0.05; fast: N=48, dt=0.12 (keşif).",
    )
    p.add_argument(
        "--hours",
        type=float,
        default=10.0,
        help="Simülasyon süresi [saat] (tavan: --max-hours).",
    )
    p.add_argument("--max-hours", type=float, default=28.0, help="Üst süre tavanı [saat].")
    p.add_argument("--cells", type=int, default=None, help="Eksen hücre sayısı N.")
    p.add_argument("--dt", type=float, default=None, help="Başlangıç zaman adımı [s] (CFL ile küçülebilir).")
    p.add_argument("--log-interval", type=int, default=1000, help="Terminal ilerlemesi her N adımda.")
    p.add_argument("--trend-sample-min", type=float, default=3.0, help="Trend PNG örnekleme [dk].")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.preset == "fast":
        n_cells, dt = 48, 0.12
    else:
        n_cells, dt = 100, 0.05
    if args.cells is not None:
        n_cells = args.cells
    if args.dt is not None:
        dt = args.dt

    run_and_visualize(
        n_cells=n_cells,
        dt=dt,
        sim_hours=args.hours,
        max_sim_hours=args.max_hours,
        log_interval=args.log_interval,
        trend_sample_min=args.trend_sample_min,
    )