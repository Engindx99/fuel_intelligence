from reporter.json_logger import log_validation


def report_validation(
    result,
    equipment,
    balance_type="energy",
):

    status = (
        "PASS"
        if result["converged"]
        else "FAIL"
    )

    validation = {
        "equipment": equipment,
        "balance_type": balance_type,
        "status": status,
        "energy_in": result["energy_in"],
        "energy_source": result["energy_source"],
        "energy_out": result["energy_out"],
        "residual": result["residual"],
        "relative_residual": result["relative_residual"],
        "absolute_tolerance": result[
            "absolute_tolerance"
        ],
        "relative_tolerance": result[
            "relative_tolerance"
        ],
    }

    log_validation(validation)

    return validation