from dataclasses import dataclass, field
import numpy as np
from typing import Dict
from chemistry.phases import SolidPhases, GasPhases
from chemistry.composition import RAW_MEAL_COMPOSITION


@dataclass
class GlobalState:

    # ======================================================
    # SIMULATION
    # ======================================================
    t: float = 0.0
    dt: float = 0.05

    # ======================================================
    # OPERATION
    # ======================================================
    Feed_rate: float = 40.0
    Fuel_rate_total: float = 3.0
    rpm: float = 0.0
    residence_time: float = 0.0
    solid_velocity: float = 0.0

    # ======================================================
    # FUEL ENERGY (W)
    # ======================================================
    Q_petcoke: float = 0.0
    Q_coal: float = 0.0
    Q_RDF: float = 0.0
    Q_H2: float = 0.0
    Q_burning: float = 0.0

    # ======================================================
    # GAS / MASS FLOW STATE
    # ======================================================
    m_dot_g: float = 0.0
    rho_g: float = 1.2

    Hgas_burning_out: float = 0.0

    Hgas_transition_in: float = 0.0
    Hgas_transition_out: float = 0.0

    Hgas_calciner_in: float = 0.0
    Hgas_calciner_out: float = 0.0

    Hgas_preheater_in: float = 0.0
    Hgas_preheater_out: float = 0.0

    Hgas_cooler_in: float = 0.0
    Hgas_cooler_out: float = 0.0
    
    # ======================================================
    # SOLID ENERGY FLOW (W)
    # ======================================================

    Hsolid_burning_out: float = 0.0

    Hsolid_transition_in: float = 0.0
    Hsolid_transition_out: float = 0.0

    Hsolid_calciner_in: float = 0.0
    Hsolid_calciner_out: float = 0.0

    Hsolid_preheater_in: float = 0.0
    Hsolid_preheater_out: float = 0.0

    Hsolid_cooler_in: float = 0.0
    Hsolid_cooler_out: float = 0.0

    
    # ================= BURNING STORED ENERGY =================
    Burning_gas_stored: float = 0.0
    Burning_solid_stored: float = 0.0
    Burning_wall_stored: float = 0.0
    Burning_stored_energy_change: float = 0.0

    # ================= TRANSITION STORED ENERGY =================
    Transition_gas_stored: float = 0.0
    Transition_solid_stored: float = 0.0
    Transition_wall_stored: float = 0.0
    Transition_stored_energy_change: float = 0.0

    # ================= CALCINER STORED ENERGY =================
    Calciner_gas_stored: float = 0.0
    Calciner_solid_stored: float = 0.0
    Calciner_wall_stored: float = 0.0
    Calciner_stored_energy_change: float = 0.0

    # ================= PREHEATER STORED ENERGY =================
    Preheater_gas_stored: float = 0.0
    Preheater_solid_stored: float = 0.0
    Preheater_wall_stored: float = 0.0
    Preheater_stored_energy_change: float = 0.0

    # ================= COOLER STORED ENERGY =================
    Cooler_gas_stored: float = 0.0
    Cooler_solid_stored: float = 0.0
    Cooler_wall_stored: float = 0.0
    Cooler_stored_energy_change: float = 0.0
    

    # ======================================================
    # ZONE ENERGY BALANCE (W)
    # ======================================================
    Burning_energy_balance: float = 0.0
    Transition_energy_balance: float = 0.0
    Calciner_energy_balance: float = 0.0
    Preheater_energy_balance: float = 0.0
    Cooler_energy_balance: float = 0.0

    # ======================================================
    # WALL HEAT LOSSES (W)
    # ======================================================

    Wall_loss_burning: float = 0.0
    Wall_loss_transition: float = 0.0
    Wall_loss_calciner: float = 0.0
    Wall_loss_preheater: float = 0.0
    Wall_loss_cooler: float = 0.0

    Total_wall_loss: float = 0.0
    
    # ================= WALL LOSS DEBUG =================
    q_loss_mean_burning: float = 0.0
    A_wall_burning: float = 0.0
    V_cell_burning: float = 0.0
    N_burning: int = 0

    q_loss_mean_transition: float = 0.0
    A_wall_transition: float = 0.0
    V_cell_transition: float = 0.0
    N_transition: int = 0
    
    # ======================================================
    # REACTION ENERGY SINKS (W)
    # ======================================================

    Calciner_Q_sink: float = 0.0

    # ======================================================
    # GLOBAL MASS & ENERGY
    # ======================================================

    Total_mass: float = 0.0
    Total_enthalpy: float = 0.0

    Total_heat_input: float = 0.0
    Total_heat_output: float = 0.0

    Reaction_heat_total: float = 0.0
    Stored_energy_total: float = 0.0

    Mass_balance_error: float = 0.0
    Energy_balance_error: float = 0.0

    Global_energy_residual: float = 0.0

    
    # ======================================================
    # GAS PHASE
    # ======================================================
    Gas_flow: float = 0.0
    O2: float = 3.5  # %
    CO2: float = 0.0
    H2O: float = 0.0
    N2: float = 0.0

    # ======================================================
    # SOLID PHASE
    # ======================================================
    m_dot_s: float = 0.0

    # ======================================================
    # REACTION STATES
    # ======================================================

    alpha_calcination: float = 0.0

    CO2_generation_rate: float = 0.0

    # ======================================================
    # CHEMISTRY
    # ======================================================
    CaCO3: float = 0.0
    CaO: float = 0.0
    SiO2: float = 0.0
    Al2O3: float = 0.0
    Fe2O3: float = 0.0

    C3S: float = 0.0
    C2S: float = 0.0
    C3A: float = 0.0
    C4AF: float = 0.0

    Free_Lime: float = 0.0
    
    # ======================================================
    # OLD PREHEATER STATES
    # ======================================================

    Tg_preheater_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 573.15
    )

    Ts_preheater_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 573.15
    )

    Tw_preheater_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 523.15
    )

    
    # ======================================================
    # PREHEATER STATES (5 CELL) (KELVIN)
    # ======================================================
    Tg_preheater: np.ndarray = field(default_factory=lambda: np.ones(5) * 573.15)

    Ts_preheater: np.ndarray = field(default_factory=lambda: np.ones(5) * 573.15)

    Tw_preheater: np.ndarray = field(default_factory=lambda: np.ones(5) * 523.15)
    
    # ======================================================
    # TRANSITION STATES (5 CELL) (KELVIN)
    # ======================================================
    Tg_transition: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1650.0
    )

    Ts_transition: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1400.0
    )

    Tw_transition: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 800.0
    )
    

    # ======================================================
    # OLD TEMPERATURE STATES (5 CELL)
    # ======================================================

    Tg_transition_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1650.0
    )

    Ts_transition_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1400.0
    )

    Tw_transition_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 800.0
    )

    Tg_calciner_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1473.15
    )

    Ts_calciner_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1223.15
    )

    Tw_calciner_old: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 773.15
    )


    # ======================================================
    # CALCINER STATES (5 CELL)
    # ======================================================

    Tg_calciner: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1473.15
    )

    Ts_calciner: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1223.15
    )

    Tw_calciner: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 773.15
    )

    # ======================================================
    # BURNING STATES (5 CELL) (KELVIN)
    # ======================================================
    
    Tg_burning_old: np.ndarray = field(
    default_factory=lambda: np.ones(5) * 1773.15)

    Ts_burning_old: np.ndarray = field(
    default_factory=lambda: np.ones(5) * 1673.15)

    Tw_burning_old: np.ndarray = field(
    default_factory=lambda: np.ones(5) * 873.15)

    Tg_burning: np.ndarray = field(default_factory=lambda: np.ones(5) * (1773.15))

    Ts_burning: np.ndarray = field(default_factory=lambda: np.ones(5) * (1673.15))

    Tw_burning: np.ndarray = field(default_factory=lambda: np.ones(5) * (873.15))

    # ======================================================
    # COOLER STATES (5 CELL) (KELVIN)
    # ======================================================

    Tg_cooler_old: np.ndarray = field(default_factory=lambda: np.ones(5) * 423.15)
    Ts_cooler_old: np.ndarray = field(default_factory=lambda: np.ones(5) * 393.15)
    Tw_cooler_old: np.ndarray = field(default_factory=lambda: np.ones(5) * 353.15)


    Tg_cooler: np.ndarray = field(default_factory=lambda: np.ones(5) * (423.15))

    Ts_cooler: np.ndarray = field(default_factory=lambda: np.ones(5) * (393.15))

    Tw_cooler: np.ndarray = field(default_factory=lambda: np.ones(5) * (353.15))
    
    
    # ======================================================
    # REACTIONS
    # ======================================================

    def __post_init__(self):

        N = self.Tg_burning.size

        # ================= SOLID PHASES =================
        self.solids = SolidPhases(
            H2O=np.full(N, RAW_MEAL_COMPOSITION["H2O"]),
            CaCO3=np.full(N, RAW_MEAL_COMPOSITION["CaCO3"]),
            CaO=np.full(N, RAW_MEAL_COMPOSITION["CaO"]),
            SiO2=np.full(N, RAW_MEAL_COMPOSITION["SiO2"]),
            Al2O3=np.full(N, RAW_MEAL_COMPOSITION["Al2O3"]),
            Fe2O3=np.full(N, RAW_MEAL_COMPOSITION["Fe2O3"]),
            C2S=np.full(N, RAW_MEAL_COMPOSITION["C2S"]),
            C3S=np.full(N, RAW_MEAL_COMPOSITION["C3S"]),
            C3A=np.full(N, RAW_MEAL_COMPOSITION["C3A"]),
            C4AF=np.full(N, RAW_MEAL_COMPOSITION["C4AF"]),
        )

        # ================= GAS PHASES =================
        self.gases = GasPhases(
            CO2=np.zeros(N),
            H2O=np.zeros(N),
        )


    # ======================================================
    # LEGACY CONVERSION VARIABLES
    # (Remove after full phase migration)
    # ======================================================

    X_CaCO3_calciner: np.ndarray = field(
        default_factory=lambda: np.ones(20)
    )

    X_CaO_calciner: np.ndarray = field(
        default_factory=lambda: np.zeros(20)
    )

    X_CaCO3_feed: float = 1.0

    X_H2O: np.ndarray = field(
        default_factory=lambda: np.ones(20)
    )

    X_OH: np.ndarray = field(
        default_factory=lambda: np.zeros(20)
    )

    X_SiO2: float = 0.0

    X_C2S: float = 0.0

    X_CaO: float = 0.0

    X_Al2O3: float = 0.0

    X_Fe2O3: float = 0.0


    # ======================================================
    # REACTION HEAT SINKS
    # ======================================================

    Drying_Q_sink: float = 0.0

    Dehydroxylation_Q_sink: float = 0.0

    Calciner_Q_sink: float = 0.0

    Belite_Q_sink: float = 0.0

    Alite_Q_sink: float = 0.0

    C3A_Q_sink: float = 0.0

    C4AF_Q_sink: float = 0.0

    Reaction_Q_sink: float = 0.0


    # ======================================================
    # LEGACY MASS FLOW VARIABLES
    # (Remove after full phase migration)
    # ======================================================

    m_dot_H2O: float = 0.0

    m_dot_H2O_dehydroxylation: float = 0.0

    m_dot_CO2: float = 0.0

    m_dot_C2S: float = 0.0

    m_dot_C3S: float = 0.0

    m_dot_C3A: float = 0.0

    m_dot_C4AF: float = 0.0
    

    # ======================================================
    # API
    # ======================================================
    Inputs: Dict = field(default_factory=dict)
    Outputs: Dict = field(default_factory=dict)
