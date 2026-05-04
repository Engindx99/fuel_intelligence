import numpy as np

class TransportModel:
    def __init__(self, config):
        self.cfg = config
        self.L = float(config['kiln']['length'])
        self.D = float(config['kiln']['diameter'])
        self.S = float(config['kiln']['slope'])

    def calculate_solid_velocity(self, current_rpm=None):
        if current_rpm is None:
            current_rpm = float(self.cfg['kiln']['rpm'])
        
        # RPM etkisini biraz daha hissettirmek için 1.05 üssü (hafif non-lineer)
        v_s = (0.19 * self.D * (current_rpm**1.05) * self.S) / 60.0 
        return max(1e-4, v_s)

    def get_dynamic_filling_degree(self, current_rpm):
        """
        RPM arttıkça doluluk oranı (bed depth) azalır.
        Bu, radyasyon alışveriş alanını doğrudan etkileyecek.
        """
        base_fill = 0.10 # %10 nominal doluluk
        # RPM 2'den 3'e çıkınca doluluk oranını %15-20 civarı düşüren ampirik bağ
        dynamic_fill = base_fill * (2.0 / max(0.5, current_rpm))**0.5
        return np.clip(dynamic_fill, 0.03, 0.15)