SEARCH_SPACE = {

    "w_T": (10.0, 500.0),

    "w_F": (0.001, 20.0),

    "w_ramp": (1.0, 5000.0),

    "max_fuel_delta": (0.02, 1.0),
}

def sample_mpc_parameters(trial):

    params = {}

    for name, (low, high) in SEARCH_SPACE.items():

        params[name] = trial.suggest_float(
            name,
            low,
            high,
        )

    return params