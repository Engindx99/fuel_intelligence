# ======================================================
# RAW MEAL COMPOSITION
# Mass fractions (sum = 1.0)
# ======================================================

RAW_MEAL_COMPOSITION = {

    "H2O": 0.00,

    "Bound_H2O": 0.020,

    "CaCO3": 0.780,

    "CaO": 0.000,

    "SiO2": 0.130,
    "Al2O3": 0.040,
    "Fe2O3": 0.030,

    "C2S":0.0,
    "C3S":0.0,
    "C3A":0.0,
    "C4AF":0.0,
}


# ======================================================
# VALIDATION
# ======================================================

total = sum(RAW_MEAL_COMPOSITION.values())

if abs(total - 1.0) > 1e-6:
    raise ValueError(
        f"Raw meal composition must sum to 1.0 (current = {total:.6f})"
    )