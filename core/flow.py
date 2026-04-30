import numpy as np
from core.state import StateIdx, IDX_FAN, IDX_REACTOR, IDX_FEED, IDX_EPSILON

# -----------------------------
# GAS FLOW
# -----------------------------
def v_gas_model(fan_rpm, epsilon):
    """
    Gaz hızı modeli: Fan devri ve boşluk oranı (epsilon) ile ters orantılı.
    """
    k = 0.001
    # Bölme hatasını önlemek için epsilon alt sınırı
    eps = np.maximum(epsilon, 0.05)
    
    # Basınç kaynaklı akış (Negatif yön: Fırın çıkışından girişine doğru)
    return -(k * fan_rpm) / eps


# -----------------------------
# SOLID FLOW
# -----------------------------
def v_solid_model(reactor_rpm, feed_rate, epsilon):
    """
    Katı hızı modeli: Fırın eğimi, çapı ve doluluk oranına bağlı.
    """
    slope = 0.03
    D = 4.5
    k = 0.02

    # Yükleme sönümlemesi (Besleme hızı arttıkça hızın hafif azalması)
    load = 1.0 / (1.0 + 0.001 * feed_rate)

    # Paketleme direnci (Boşluk oranı azaldıkça sürtünme artar)
    eps = np.clip(epsilon, 0.0, 0.9)
    packing = (1.0 - eps)**1.5

    return k * D * slope * reactor_rpm * load * packing


# -----------------------------
# VELOCITY FIELD
# -----------------------------
def compute_velocities(x, u):
    """
    Mevcut state (x) ve kontrol (u) vektörlerinden faz hızlarını hesaplar.
    """
    fan = u[IDX_FAN]
    reactor = u[IDX_REACTOR]
    feed = u[IDX_FEED]
    epsilon = x[IDX_EPSILON]

    v_g = v_gas_model(fan, epsilon)
    v_s = v_solid_model(reactor, feed, epsilon)

    return v_s, v_g


# -----------------------------
# POROSITY DYNAMICS
# -----------------------------
def compute_porosity(x, u):
    """
    Kalsinasyon reaksiyonuna bağlı gözeneklilik (porosity) değişimi.
    """
    # İndeksleri StateIdx üzerinden güvenli alıyoruz
    CaCO3 = x[StateIdx.CaCO3]
    CaO = x[StateIdx.CaO]

    # Reaksiyon kaynaklı genleşme katsayısı
    k = 0.05

    # CaCO3 azaldıkça ve CaO oluştukça gözeneklilik karakteristiği değişir
    return k * np.maximum(CaCO3, 0.0) * (1.0 - np.maximum(CaO, 0.0))