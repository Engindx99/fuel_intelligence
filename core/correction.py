import numpy as np

def mass_drift_correction(X, mw, target_mass=None, strength=1.0):
    """
    Physics-preserving mass drift correction.

    Parameters
    ----------
    X : (N_cells, N_species)
    mw : (N_species,)
    target_mass : (N_cells,) or None
        If None → keep current total mass (closed system assumption)
    strength : float (0-1)
        1.0 = full correction
        <1.0 = partial correction (safer for stiff kinetics)
    """

    # -----------------------------
    # current total mass per cell
    # -----------------------------
    current_mass = np.sum(X * mw, axis=1)  # (N_cells,)

    # -----------------------------
    # define target
    # -----------------------------
    if target_mass is None:
        target_mass = current_mass  # closed system assumption

    # -----------------------------
    # residual
    # -----------------------------
    mass_error = target_mass - current_mass  # (N_cells,)

    mw_safe = mw + 1e-12

    # -----------------------------
    # distribute correction proportionally
    # -----------------------------
    species_mass = X * mw  # mass contribution per species

    species_share = species_mass / (np.sum(species_mass, axis=1, keepdims=True) + 1e-12)

    correction_mass = mass_error[:, None] * species_share

    correction_X = correction_mass / mw_safe

    # -----------------------------
    # apply correction (relaxed)
    # -----------------------------
    X_corrected = X + strength * correction_X

    # enforce non-negativity
    X_corrected = np.maximum(X_corrected, 0.0)

    return X_corrected