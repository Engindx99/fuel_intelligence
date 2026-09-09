def report_validation(result, equipment, balance_type="energy"):
    status = "PASS" if result["converged"] else "FAIL"

    print(
        f"[{status}] "
        f"{equipment} {balance_type} balance | "
        f"residual={result['residual']:.6e} W | "
        f"relative={result['relative_residual']:.6e}"
    )