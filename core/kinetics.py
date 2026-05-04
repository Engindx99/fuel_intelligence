import numpy as np
from numba import njit

@njit
def compute_reaction_rate_numba(T, X, k0, Ea, R, T_min):
    """C hızında çalışan Arrhenius kalsinasyon hızı."""
    if T < T_min or X >= 1.0:
        return 0.0
    
    # k = k0 * exp(-Ea / (R * T))
    k = k0 * np.exp(-Ea / (R * T))
    
    # r = k * (1-X)
    rate = k * (1.0 - X)
    return min(rate, 0.01)

class CalcinationKinetics:
    """Simulation scriptinin beklediği sınıf ismi."""
    def __init__(self, config):
        # YAML'dan parametreleri yükle
        self.k0 = float(config['kinetics']['k0'])
        self.Ea = float(config['kinetics']['Ea'])
        self.R = float(config['kinetics']['R'])
        self.T_min = float(config['kinetics']['T_min_rxn'])
        self.dH = float(config['kinetics']['dH'])

    def compute_rate(self, T, X):
        """Python tarafında çağrılırsa Numba fonksiyonuna yönlendirir."""
        return compute_reaction_rate_numba(T, X, self.k0, self.Ea, self.R, self.T_min)