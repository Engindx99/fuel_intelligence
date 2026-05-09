import numpy as np

def safe_f(value):
    if isinstance(value, list):
        return float(value[0])
    return float(value)


class EnergyModel:
    def __init__(self, config):
  
        self.base_h_gs = safe_f(config['gas']['h_gs'])

        self.sigma = 5.67037e-8  # W/(m^2.K^4)

        self.eps_g = safe_f(config['gas'].get('emissivity_g'))
        

        self.eps_s = safe_f(config['material'].get('emissivity_s'))
        

        self.eps = 0.5 * (self.eps_g + self.eps_s)
        

    def calculate_convection_coeff(self,current_fan_rate,nominal_fan=800.0):

        fan_ratio = current_fan_rate / nominal_fan

        return self.base_h_gs * (fan_ratio ** 0.4)
    

    def calculate_radiation_flux(self, Tg, Ts, area):

        return self.eps * self.sigma * area * (Tg**4 - Ts**4)    
            


    def get_reaction_heat(self,rates,m_dot_s,dH_vec):

        return np.sum(rates * dH_vec) * m_dot_s