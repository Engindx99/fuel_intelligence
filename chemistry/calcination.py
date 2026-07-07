rate = self.reaction_rate(
    state.Ts_calciner
)

reacted = self.reacted_mass(
    state.solids.CaCO3,
    rate,
    state.dt,
)

state.solids.CaCO3 -= reacted

state.solids.CaO += reacted

state.gases.CO2 += reacted

state.Calciner_Q_sink = np.sum(
    self.heat_sink(
        reacted,
    )
)