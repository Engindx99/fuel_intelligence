import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_kiln_results(z, Ts, Tg, X, title="Rotary Kiln Calcination Profile"):
    """
    Döner fırın eksenel profillerini görselleştirir (Mevcut kod - Bozulmadı).
    """
    plt.ioff() 
    plt.close('all')
    plt.rcParams['figure.facecolor'] = 'white'
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Axial Distance (z) [m]', fontsize=12)
    ax1.set_ylabel('Temperature [K]', color='tab:red', fontsize=12)
    
    z = np.array(z, dtype=float)
    Ts = np.array(Ts, dtype=float)
    Tg = np.array(Tg, dtype=float)
    X = np.array(X, dtype=float)

    line1 = ax1.plot(z, Tg, 'r-', linewidth=2.5, label='Gas Temp (Tg) [Flame End -> Feed]')
    line2 = ax1.plot(z, Ts, 'b-', linewidth=2.5, label='Solid Temp (Ts) [Feed -> Discharge]')
    
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Conversion (X) [0-1]', color='tab:green', fontsize=12)
    line3 = ax2.plot(z, X, 'g--', linewidth=2, label='CaCO3 Conversion (X)')
    
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(-0.02, 1.02)

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower center', bbox_to_anchor=(0.5, -0.18), 
                ncol=3, frameon=True, shadow=True)

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    print("\n📊 Eksenel profil grafiği gösteriliyor.", flush=True)
    plt.show()

def plot_time_series(history, title="Rotary Kiln Time-Series Analysis"):
    """
    Simülasyon boyunca kaydedilen zaman serisi verilerini görselleştirir.
    """
    if not history or 'time' not in history:
        print("⚠️ Hata: Zaman serisi verisi bulunamadı.")
        return

    plt.ioff()
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Zaman ekseni ve Sıcaklıklar
    t = np.array(history['time'])
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Avg Temperature [K]', color='tab:red', fontsize=12)
    
    l1 = ax1.plot(t, history['avg_Tg'], 'r-', label='Avg Gas Temp (Tg)')
    l2 = ax1.plot(t, history['avg_Ts'], 'b-', label='Avg Solid Temp (Ts)')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Sağ eksen: Maksimum Dönüşüm
    ax2 = ax1.twinx()
    ax2.set_ylabel('Max Conversion (X)', color='tab:green', fontsize=12)
    l3 = ax2.plot(t, history['max_X'], 'g--', label='Max System Conversion')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(-0.05, 1.05)

    # Legend
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best', frameon=True)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    print("📈 Zaman serisi grafiği gösteriliyor.", flush=True)
    plt.show()

# --- TEST BÖLÜMÜ ---
if __name__ == "__main__":
    print("🛠 plots.py test modunda çalışıyor...")
    
    # Eksenel Test
    z_test = np.linspace(0, 60, 100)
    Ts_test = 300 + 900 * (1 - np.exp(-z_test/25))
    Tg_test = 1400 - 800 * (z_test/60)**0.5
    X_test = np.clip((Ts_test - 1000)/200, 0, 1)
    plot_kiln_results(z_test, Ts_test, Tg_test, X_test, title="DEBUG: Axial Profile Test")

    # Zaman Serisi Test
    history_test = {
        'time': np.arange(0, 1000, 10),
        'avg_Ts': 300 + 800 * (1 - np.exp(-np.arange(0, 1000, 10)/200)),
        'avg_Tg': 1200 + 100 * np.sin(np.arange(0, 1000, 10)/50),
        'max_X': np.clip(np.arange(0, 1000, 10)/800, 0, 1)
    }
    plot_time_series(history_test)