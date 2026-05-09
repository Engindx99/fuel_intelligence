import numpy as np

def safe_f(value):
    if isinstance(value, list): return float(value[0])
    return float(value)

class EnergyModel:
    def __init__(self, config):
        self.base_h_gs = safe_f(config['gas']['h_gs']) 
        self.sigma = 5.67037e-8
        self.eps_g = safe_f(config['gas'].get('emissivity_g', 0.3))
        self.eps_s = safe_f(config['material'].get('emissivity_s', 0.85))
        self.eps_eff = (self.eps_g + self.eps_s) / 2.0
        
        # Fiziksel Dağılım Katsayıları (Energy Partitioning)
    
        self.f_solid = 0.001 
        self.f_wall_loss = 0.799 
        self.f_gas_internal = 0.200 

    def calculate_convection_coeff(self, current_fan_rate, nominal_fan=850.0):
        fan_ratio = current_fan_rate / nominal_fan
        return self.base_h_gs * (fan_ratio**0.4)

    def calculate_radiation_distribution(self, Tg, Ts, Tw, area):

        # Teorik maksimum transfer potansiyeli
        q_max = self.eps_eff * self.sigma * area * (Tg**4 - Ts**4)
        
        # 1. Katı fazın aldığı net ısı (Senin istediğin kısıtlanmış miktar)
        q_to_solid = q_max * self.f_solid
        
        # 2. Duvar ve dış ortam kayıpları (Isıl dengede kayıp olarak tanımlanır)
        q_to_wall = q_max * self.f_wall_loss
        
        # 3. Gaz fazının geri soğurduğu veya sistemde dağılan miktar
        q_internal_gas = q_max * self.f_gas_internal
        
        return q_to_solid, q_to_wall, q_internal_gas

    def get_reaction_heat(self, rates, m_dot_s, dH_vec):
        return np.sum(rates * dH_vec) * m_dot_s