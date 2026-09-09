def validate_energy(
    energy_in,
    energy_out,
    absolute_tolerance=1e-3,
    relative_tolerance=1e-9,
):
    residual = energy_in - energy_out

    scale = max(
        abs(energy_in),
        abs(energy_out),
        1.0,
    )

    relative_residual = abs(residual) / scale

    converged = (
        abs(residual) <= absolute_tolerance
        or relative_residual <= relative_tolerance
    )

    return {
        "energy_in": float(energy_in),
        "energy_out": float(energy_out),
        "residual": float(residual),
        "relative_residual": float(relative_residual),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "converged": converged,
    }