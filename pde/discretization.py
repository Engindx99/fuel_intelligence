# pde/discretization.py

"""
Spatial discretization module for rotary kiln PDE.

Implements:
- Finite Difference Method (FDM)
- Upwind scheme for counter-current flow
- Spatial derivative calculation (dx/dz)

NO:
- Time integration (handled in simulation)
- Reaction physics (handled in model)
"""

import numpy as np
from core.state import *

def compute_spatial_derivatives(X, v_s, v_g, dz):
    """
    Computes dx/dz for all states across the kiln length.
    
    X    : State matrix [N_cells, N_states]
    v_s  : Solid velocity (scalar or vector)
    v_g  : Gas velocity (scalar or vector)
    dz   : Step size (Kiln_Length / N_cells)
    """
    
    N_cells = X.shape[0]
    dx_dz = np.zeros_like(X)

    # 1. SOLID PHASE & TEMPERATURE (Forward Flow)
    # -------------------------------------------
    # For v_s > 0, we look 'backwards' (i - i-1)
    for i in range(N_cells):
        if i == 0:
            # Boundary Condition: Left side (Inlet)
            # Typically dx/dz is 0 or handled by specific inlet state
            dx_dz[i, SOLID_SPECIES + [IDX_T_S]] = 0.0
        else:
            dx_dz[i, SOLID_SPECIES + [IDX_T_S]] = (X[i] - X[i-1])[SOLID_SPECIES + [IDX_T_S]] / dz

    # 2. GAS PHASE & TEMPERATURE (Backward Flow)
    # -------------------------------------------
    # For v_g < 0 (counter-current), we look 'forward' (i+1 - i)
    for i in range(N_cells):
        if i == N_cells - 1:
            # Boundary Condition: Right side (Burner/Gas Inlet)
            dx_dz[i, GAS_SPECIES + [IDX_T_G]] = 0.0
        else:
            # Note: v_g is already negative in flow.py, 
            # the logic here handles the spatial gradient direction.
            dx_dz[i, GAS_SPECIES + [IDX_T_G]] = (X[i+1] - X[i])[GAS_SPECIES + [IDX_T_G]] / dz

    return dx_dz