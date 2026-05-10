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
        self.S = float(self._safe_f(config['kiln']['slope'])) # Örn: 0.03 (eğim)

    def _safe_f(self, v):
        """Config dosyasından gelen liste veya tekil değerleri float'a çevirir."""
        return float(v[0]) if isinstance(v, list) else float(v)

    def calculate_solid_velocity(self, current_rpm=None):
        """
        Malzemenin fırın ekseni boyunca ilerleme hızını (m/s) hesaplar.
        Sullivan (USBM) formülü temel alınmıştır.
        """
        if current_rpm is None:
            current_rpm = float(self._safe_f(self.cfg['kiln']['rpm']))
        
        # v_s_min_sec = (k * RPM * D * S) / 60
        # Katsayı 1.77: Malzemenin fırında ~60-70 dakika kalmasını sağlar (3 RPM için).
        v_s_min_sec = (1.77 * current_rpm * self.D * self.S) / 60.0
        
        # ALT LİMİT KORUMASI: 0.020 çok hızlıydı ve %1 fCaO'ya izin vermiyordu.
        # 0.012'ye çekerek malzemenin fırın sonunda daha fazla "pişmesine" izin veriyoruz.
        return max(0.018, v_s_min_sec)

    def get_dynamic_filling_degree(self, current_rpm):
        """
        Fırın doluluk oranı (Filling Degree / Bed Depth).
        RPM azaldıkça yatak derinliği artar, bu da termal ataleti yükseltir.
        """
        if current_rpm is None:
            current_rpm = float(self._safe_f(self.cfg['kiln']['rpm']))
            
        # Nominal %11 doluluk hedefi
        base_fill = 0.11 
        
        # RPM etkisi: RPM düştükçe yatak derinliği doğrusal olmayan şekilde artar.
        # Bu değer solver.py içinde ısı transfer alanını (q_gs_vec) etkiler.
        dynamic_fill = base_fill * (3.0 / max(0.5, current_rpm))**0.6
        
        # İşletme limitleri: %5 ile %18 arası
        return np.clip(dynamic_fill, 0.05, 0.18)

    def calculate_residence_time(self, current_rpm):
        """
        Malzemenin fırın içindeki toplam kalma süresini (dakika) döndürür.
        """
        v_s = self.calculate_solid_velocity(current_rpm)
        time_sec = self.L / v_s
        return time_sec / 60.0