import numpy as np

class CalcinationModel:

    def __init__(self, N=5, dz=1.0):

        self.N = N
        self.dz = dz

        self.R = 8.314462618
        self.eps = 1e-9

        # ======================================================
        # CALCINATION KINETICS AND ENTHALPY
        # ======================================================
        self.A = 1.5e4          # 1/s
        self.Ea = 1.8e5         # J/mol

        # CaCO3 decomposition heat
        self.deltaH_calc = 178000.0   # J/kg CaCO3

        # ======================================================
        # MOLAR MASSES
        # ======================================================
        self.M_CaCO3 = 100.09   # kg/kmol
        self.M_CO2 = 44.01      # kg/kmol

        self.CO2_ratio = (self.M_CO2 / self.M_CaCO3)
            
     
        # ======================================================
        # WORK ARRAYS
        # ======================================================
        self._dX_dz = np.zeros(N)

    # ======================================================
    # MAIN ENTRY
    # ======================================================
    def apply(self, state, dt):

        self.calcination(state, dt)
 
        return state

    # ======================================================
    # CALCINATION
    # ======================================================
    def calcination(self, state, dt):

        # ======================================================
        # TEMPERATURE FIELD
        # ======================================================
        Ts = state.Ts_calciner

        # ======================================================
        # SOLID VELOCITY
        # ======================================================
        u_s = state.u_s

        # ======================================================
        # CaCO3 REMAINING FRACTION
        # ======================================================
        X = state.X_CaCO3_calciner

        # ======================================================
        # SOLVE CONVERSION PDE
        # ======================================================
        X_new, reaction_rate = self.solve_conversion(
            X=X,
            Ts=Ts,
            u_s=u_s,
            dt=dt,
            X_feed=state.X_CaCO3_feed,
        )

        # ======================================================
        # UPDATE COMPOSITION
        # ======================================================

        # Remaining CaCO3 fraction
        state.X_CaCO3_calciner = X_new

        # Converted fraction to CaO
        state.X_CaO_calciner = 1.0 - X_new


        # ======================================================
        # REACTION HEAT SINK
        # ======================================================
        state.Calciner_Q_sink = self.calcination_heat_sink(state=state, reaction_rate=reaction_rate,)
            

        # ======================================================
        # CO2 GENERATION
        # ======================================================
        state.m_dot_CO2_calciner = self.calcination_CO2(state, reaction_rate)
  

    # ======================================================
    # CALCINATION RATE
    # ======================================================
    def calcination_rate(self, Ts, X):

        k = self.A * np.exp(-self.Ea / (self.R * (Ts + self.eps)))

        r = k * X

        return r

    # ======================================================
    # CONVERSION PDE
    # ======================================================
    def solve_conversion(self, X, Ts, u_s, dt, X_feed):

        # ======================================================
        # GRADIENT
        # ======================================================
        dX_dz = self._dX_dz

        dX_dz[1:] = (X[1:] - X[:-1]) / self.dz

        dX_dz[0] = dX_dz[1]

        # ======================================================
        # REACTION RATE
        # ======================================================
        reaction_rate = self.calcination_rate(Ts, X)

        # ======================================================
        # PREVENT OVERSHOOT
        # ======================================================
        reaction_rate = np.minimum(reaction_rate, X / (dt + self.eps),)
            
        # ======================================================
        # CONVERSION EQUATION
        # ======================================================
        X_new = X + dt * (-u_s * dX_dz - reaction_rate)
        
        # ======================================================
        # INLET BOUNDARY
        # ======================================================
        X_new[0] = X_feed

        # ======================================================
        # LIMITS
        # ======================================================
        np.clip(X_new, 0.0, 1.0, out=X_new)


        return X_new, reaction_rate
    
    # ======================================================
    # CALCINATION HEAT SINK
    # ======================================================
    def calcination_heat_sink(self, state, reaction_rate):

        # ======================================================
        # CaCO3 MASS FLOW
        # ======================================================
        m_dot_CaCO3 = (state.m_dot_s * state.CaCO3_mass_fraction)
            

        # ======================================================
        # AVERAGE REACTION RATE
        # ======================================================
        r_mean = np.mean(reaction_rate)
            
        # ======================================================
        # REACTED CaCO3 MASS FLOW
        # ======================================================
        m_dot_reacted = (m_dot_CaCO3 * r_mean)

        
        # ======================================================
        # HEAT CONSUMPTION
        # ======================================================
        Q_sink = (
            m_dot_reacted
            *
            self.deltaH_calc
        )

        return Q_sink

    # ======================================================
    # CO2 GENERATION
    # ======================================================
    def calcination_CO2(self, state, reaction_rate):

        # ======================================================
        # CaCO3 MASS FLOW
        # ======================================================
        m_dot_CaCO3 = (state.m_dot_s * state.CaCO3_mass_fraction)
            

        # ======================================================
        # AVERAGE REACTION RATE
        # ======================================================
        r_mean = np.mean(reaction_rate)
   
        # ======================================================
        # REACTED CaCO3 MASS FLOW
        # ======================================================
        m_dot_reacted = (m_dot_CaCO3 * r_mean)
        

        # ======================================================
        # CO2 MASS FLOW
        # ======================================================
        m_dot_CO2 = (m_dot_reacted * self.CO2_ratio)
            
        return m_dot_CO2
    
    
    
