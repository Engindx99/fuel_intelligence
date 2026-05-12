import numpy as np

class TransportModel:
    def __init__(self, config):
        """
        Döner fırın taşıma modeli - Fiziksel korelasyonlar ve kütle akış yönetimi.
        """
        # Config hiyerarşisine güvenli erişim
        kiln_cfg = config.get("kiln", {})
        
        self.L = float(kiln_cfg.get("length", 60.0))    #
        self.D = float(kiln_cfg.get("diameter", 4.2))  #
        self.S = float(kiln_cfg.get("slope", 0.03))    #

    def get_dynamic_filling_degree(self, current_rpm, feed_rate_ton_h=None):
        """
        Fırın doluluk oranını (filling degree) hesaplar. 
        """
        rpm = max(0.5, float(current_rpm))
        base_fill = 0.10 # de tanımlanan nominal değer

        # RPM arttıkça malzeme daha hızlı akar, doluluk oranı azalır
        dynamic_fill = base_fill * (2.0 / rpm) ** 0.35 # 2.0 RPM baz alındı

        if feed_rate_ton_h is not None:
            # Besleme hızı (ton/h) arttıkça yatak derinliği artar
            feed_factor = (feed_rate_ton_h / 125.0) ** 0.2
            dynamic_fill *= feed_factor

        return np.clip(dynamic_fill, 0.05, 0.20)

    def calculate_solid_velocity(self, current_rpm, fill_degree=None):
        """
        Sullivan tipi korelasyon ile eksenel katı hızını [m/s] hesaplar.
        """
        rpm = max(0.1, float(current_rpm))
        
        # Sullivan korelasyonu (Temel hız hesaplaması)
        # v_base = (1.77 * rpm * D * S) / 60 [m/s]
        v_base = (1.77 * rpm * self.D * self.S) / 60.0

        # Doluluk oranı düzeltmesi: 
        # Daha derin yatak, sürtünme nedeniyle eksenel hızı bir miktar sınırlar
        if fill_degree is not None:
            # Nominal 0.10 doluluk oranına göre ölçekleme
            fill_correction = (0.10 / max(fill_degree, 0.05)) ** 0.5
            v_s = v_base * fill_correction
        else:
            v_s = v_base

        # Nümerik kararlılık için alt sınır
        return max(0.001, v_s)

    def check_cfl_condition(self, v_s, dt, nodes):
        """
        CFL Kararlılık Kontrolü: dt <= dz / v_s
        """
        dz = self.L / nodes
        max_dt = dz / max(v_s, 1e-6)
        
        if dt > max_dt:
            # Yapay nümerik salınım riski
            return False, max_dt
        return True, max_dt