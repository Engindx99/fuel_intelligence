import numpy as np

class TransportModel:
    def __init__(self, config):
        """
        Döner fırın malzeme taşınım modeli.
        Birimler: L [m], D [m], S [oran, m/m]
        """
        self.cfg = config
        self.L = float(self._safe_f(config['kiln']['length']))
        self.D = float(self._safe_f(config['kiln']['diameter']))
        self.S = float(self._safe_f(config['kiln']['slope'])) # Örn: 0.03 (derece değil eğim)

    def _safe_f(self, v):
        return float(v[0]) if isinstance(v, list) else float(v)

    def calculate_solid_velocity(self, current_rpm=None):
        """
        Malzemenin fırın ekseni boyunca ilerleme hızını (m/s) hesaplar.
        Loglardaki 1600 tonluk yığılmayı engellemek için hız katsayısı 
        literatürdeki 1.77 * factor değerine göre optimize edilmiştir.
        """
        if current_rpm is None:
            current_rpm = float(self._safe_f(self.cfg['kiln']['rpm']))
        
        # Sullivan (USBM) formülü modifiyeli: 
        # v_s_min = (k * RPM * D * S) / 60
        # 125 t/h için ideal v_s aralığı: 0.022 - 0.035 m/s
        
        # Katsayı (1.85), malzemenin fırındaki kalma süresini (residence time) 
        # ~40-50 dakika bandına çeker.
        v_s_min_sec = (1.85 * current_rpm * self.D * self.S) / 60.0
        
        # Alt limit koruması: Malzeme çok yavaşlarsa (RPM düşerse) 
        # hold-up'ın sonsuza gitmesini engeller.
        return max(0.020, v_s_min_sec)

    def get_dynamic_filling_degree(self, current_rpm):
        """
        Fırın doluluk oranı (Filling Degree / Bed Depth).
        RPM ile ters orantılı olarak değişir, radyasyon alanını etkiler.
        """
        if current_rpm is None:
            current_rpm = float(self._safe_f(self.cfg['kiln']['rpm']))
            
        # Nominal %8-10 doluluk hedefi
        base_fill = 0.10 
        
        # RPM arttıkça yatak yüksekliği azalır (Hız arttığı için)
        dynamic_fill = base_fill * (2.5 / max(0.5, current_rpm))**0.5
        
        # İşletme limitleri: %4 ile %15 arası
        return np.clip(dynamic_fill, 0.04, 0.15)

    def calculate_residence_time(self, current_rpm):
        """
        Malzemenin fırın içindeki toplam kalma süresini (dakika) döndürür.
        Bilgi amaçlıdır.
        """
        v_s = self.calculate_solid_velocity(current_rpm)
        time_sec = self.L / v_s
        return time_sec / 60.0