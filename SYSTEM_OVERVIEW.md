# SYSTEM OVERVIEW

This system is a physics-based rotary kiln digital twin.

## CORE COMPONENTS

1. State Manager
2. Energy Balance Engine
3. Reaction Engine
4. Transport (RTD / convection)
5. Control Interface (MPC-ready)

## EXECUTION LOOP

1. Read control inputs u
2. Compute Q_fuel from LHV
3. Compute reaction rates
4. Compute heat transfer
5. Update states via dx/dt
6. Enforce constraints
7. Output state

## DESIGN GOALS

- Deterministic behavior
- Physics consistency
- MPC compatibility