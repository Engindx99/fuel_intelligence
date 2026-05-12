import numpy as np

class TransportModel:
    def __init__(self, config):
        self.cfg = config
        self.L = self._safe_f(config["kiln"]["length"])
        self.D = self._safe_f(config["kiln"]["diameter"])
        self.S = self._safe_f(config["kiln"]["slope"])

    def _safe_f(self, value):
        if isinstance(value, list):
            return float(value[0])
        return float(value)

    def get_dynamic_filling_degree(self, current_rpm, feed_rate=None):
        """
        Fırın doluluk oranını (filling degree) hesaplar. 
        Solver'da hız hesaplamasından önce çağrılmalıdır.
        """
        rpm = max(0.5, float(current_rpm))
        base_fill = 0.12 # Nominal doluluk oranı [%]

        # RPM arttıkça doluluk oranı azalır (hız arttığı için)
        dynamic_fill = base_fill * (3.5 / rpm) ** 0.35

        if feed_rate is not None:
            # Besleme hızı arttıkça yatak derinleşir
            feed_factor = (feed_rate / 125.0) ** 0.15
            dynamic_fill *= feed_factor

        return np.clip(dynamic_fill, 0.06, 0.18)

    def calculate_solid_velocity(self, current_rpm, fill_degree=None):
        """
        Sullivan tipi korelasyon ile eksenel katı hızını [m/s] hesaplar.
        """
        rpm = max(0.1, float(current_rpm))
        
        # Sullivan korelasyonu (Temel hız)
        # v_base = (1.77 * rpm * D * S) / 60 [m/s]
        v_base = (1.77 * rpm * self.D * self.S) / 60.0

        # Eğer doluluk oranı verilmişse, yatak derinliği düzeltmesi uygula
        # Daha derin yatak (yüksek fill_degree) sürtünme ve içsel akış nedeniyle hızı bir miktar düşürür
        if fill_degree is not None:
            fill_correction = (0.12 / max(fill_degree, 0.05)) ** 0.5
            v_s = v_base * fill_correction
        else:
            v_s = v_base

        return max(0.002, v_s)

    def calculate_residence_time(self, current_rpm, fill_degree=0.12):
        """
        Ortalama kalış süresi [dakika].
        """
        v_s = self.calculate_solid_velocity(current_rpm, fill_degree)
        time_sec = self.L / v_s
        return time_sec / 60.0