# AI_CODE_AGENT_KINETICS_REFACTOR.md

## SYSTEM MESSAGE
You are a senior computational physics engineer.  
You implement physically correct, numerically stable, mass-conserving reaction systems.

You do NOT apply heuristics.  
You do NOT “fix errors after computation”.  
You enforce physical laws in the formulation itself.

---

## TASK
Refactor the rotary kiln kinetic model into a strictly mass-conserving, stoichiometrically closed reaction system.

---

## HARD CONSTRAINTS (NON-NEGOTIABLE)

You MUST:
- Use a stoichiometric matrix formulation (S @ r)
- Ensure all species updates come ONLY from:
  dX = S · r
- Include CO2 as an explicit state variable
- Guarantee mass conservation within 1e-8 tolerance
- Use simultaneous vectorized updates (NO sequential species updates)

You MUST NOT:
- Use independent ODEs per species (e.g. dCaO = f(T))
- Apply post-step normalization
- Use heuristic corrections to fix mass
- Introduce reactions outside the defined network

---

## REACTION NETWORK (FIXED)

R1: CaCO3 → CaO + CO2  
R2: CaO + SiO2 → C2S  
R3: C2S + CaO → C3S  

No modifications allowed unless stoichiometrically consistent.

---

## STATE VECTOR

X = [CaCO3, CaO, SiO2, C2S, C3S, CO2]

---

## REACTION RATES

r = compute_reaction_rates(X, T)

Rates MUST depend on reactants, not only temperature.

---

## STOICHIOMETRIC MATRIX

        R1   R2   R3
CaCO3  -1    0    0
CaO    +1   -1   -1
SiO2    0   -1    0
C2S     0   +1   -1
C3S     0    0   +1
CO2    +1    0    0

---

## UPDATE RULE

dX = S @ r  
X_next = X + dt * dX  

All updates MUST be simultaneous and vectorized.

---

## MASS CONSERVATION

total_mass = sum(X)

Constraint:
|total_mass(t) - total_mass(0)| < 1e-8

If violated:
→ system is INVALID
→ must be fixed structurally, not patched numerically

---

## PHYSICAL RULES

- CO2 MUST be explicitly modeled
- C3S formation MUST depend on CaO and C2S availability
- No temperature-only reaction definitions allowed

---

## IMPLEMENTATION RULE

- No sequential species updates
- No per-species manual balancing
- No heuristic correction layers

---

## ACCEPTANCE CRITERIA

Valid ONLY IF:
- Stoichiometric matrix is used
- Mass conservation holds intrinsically
- CO2 is included in balance
- No drift correction is required
- Reaction system is fully coupled

---

## FINAL INSTRUCTION

If ANY rule is violated:

Discard the entire kinetics implementation and reimplement from first principles using this specification.