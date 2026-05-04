import numpy as np

def safe_f(value):
    if isinstance(value, list): return float(value[0])
    return float(value)

class CalcinationKinetics:
    def __init__(self, config):
        self.k0 = safe_f(config['kinetics']['k0'])
        self.Ea = safe_f(config['kinetics']['Ea'])
        self.R = safe_f(config['kinetics']['R'])
        self.dH = safe_f(config['material']['dh_rxn'])
        self.T_min = safe_f(config['kinetics'].get('T_min_rxn', 1073.0))

    def compute_rate(self, Ts, X):
        # Eksi işareti (unary -) burada hata veriyor olabilir, float zorlaması yaptık
        if float(Ts) < self.T_min:
            return 0.0
        
        # Arrhenius hızı
        k = self.k0 * np.exp(-self.Ea / (self.R * float(Ts)))
        rate = k * np.power(max(0.0, 1.0 - float(X)), 2/3)
        return rate