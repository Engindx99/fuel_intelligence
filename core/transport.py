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
        Sullivan formülüyle hız hesaplama.
        Saturasyonu aşmak için malzemenin fırında kalma süresini (residence time) 
        optimize eden hız limitleri uygulandı.
        """
        if current_rpm is None:
            current_rpm = float(self._safe_f(self.cfg['kiln']['rpm']))
        
        # Sullivan katsayısı (1.77) üzerinden ham hız
        v_s_raw = (1.77 * max(0.1, current_rpm) * self.D * self.S) / 60.0
        
        # ANALİZ: 1063K saturasyonu, malzemenin kalsinasyon bölgesinden 
        # "enerjisini tam almadan" kaçtığını gösteriyor olabilir.
        # Üst limiti 0.015'e çekerek aşırı hızlı akışı frenliyoruz.
        return np.clip(v_s_raw, 0.007, 0.015)

    def get_dynamic_filling_degree(self, current_rpm):
        """
        Fırın doluluk oranı. 
        Saturasyonu kırmak için yatak derinliği (doluluk) optimize edildi.
        """
        if current_rpm is None:
            current_rpm = float(self._safe_f(self.cfg['kiln']['rpm']))
            
        # Baz doluluk oranı %12 (İdeal ısı transfer alanı sağlar)
        base_fill = 0.12 
        
        # RPM etkisi: RPM düştükçe yatak derinleşir (Atalet artar).
        # Saturasyon varsa, yatağın çok derinleşip ısıyı çekirdeğe iletememesinden kaçınmalıyız.
        # Bu yüzden üst limit %15'e çekildi.
        nominal_rpm = 3.5
        dynamic_fill = base_fill * (nominal_rpm / max(0.5, current_rpm))**0.4
        
        return np.clip(dynamic_fill, 0.08, 0.15)

    def calculate_residence_time(self, current_rpm):
        """
        Malzemenin fırın içindeki toplam kalma süresini (dakika) döndürür.
        """
        v_s = self.calculate_solid_velocity(current_rpm)
        time_sec = self.L / v_s
        return time_sec / 60.0