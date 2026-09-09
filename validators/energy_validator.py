def validate_energy(
    energy_in,
    energy_out,
    tolerance=1e-3,
):
    residual = energy_in - energy_out

    scale = max(
        abs(energy_in),
        abs(energy_out),
        1.0,
    )

    relative_residual = abs(residual) / scale

    return {
        "energy_in": float(energy_in),
        "energy_out": float(energy_out),
        "residual": float(residual),
        "relative_residual": float(relative_residual),
        "tolerance": float(tolerance),
        "converged": abs(residual) <= tolerance,
    }