# ROTARY KILN MODEL SPEC (LHV-DEPENDENT)

## STATES

x = [
  T_g[i],     # gas temperature
  T_s[i],     # solid temperature
  Y_g[i,k],   # gas species fractions
  X_s[i,k]    # solid species fractions
]

## CONTROLS

u = [
  fuel_rate,
  fan_rpm,
  kiln_rpm,
  feed_rate
]

## ENERGY BALANCE (GAS)

dT_g/dt =
  (Q_fuel + Q_reaction - Q_gs - Q_loss) / (m_g * Cp_g)

## ENERGY BALANCE (SOLID)

dT_s/dt =
  (Q_gs + Q_reaction_solid) / (m_s * Cp_s)

## HEAT TRANSFER

Q_gs = h * A * (T_g - T_s)

## FUEL INPUT

Q_fuel = fuel_rate * LHV

## REACTION KINETICS

k(T) = A * exp(-Ea / (R * T))

## MASS CONSERVATION

Sum(Y_g) = 1
Sum(X_s) = 1

## SPATIAL MODEL

Discretize kiln into N cells

For each cell i:
- exchange with i-1 and i+1
- include convection + diffusion terms

## STATE-SPACE FORM

dx/dt = f(x, u, θ)

y = g(x)